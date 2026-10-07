/* TRGoChessC-inspired teacher: the original 16 patterns and G11 weights.
 * Reimplemented search with explicit bounds/depth, safe indices and per-board state.
 * OpenMP parallelizes independent boards, never shares mutable search state. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define MAX_N 25
#define CELLS (MAX_N * MAX_N)
#define LINES (6 * MAX_N)
#define DFA_N 512
#define TT_N 16384
#define WIN 1e12
#define INF 1e30

static const char *patterns[16] = {
    "210", "010", "2110", "0110", "21110", "01110", "21010", "01010",
    "211010", "210110", "010110", "211110", "011110", "10111", "11011", "11111"
};
static const double weights[16] = {
    1.147176, 1.017057, 2.872098, 3.774109, 14.642408, 39.582565, .1, .1,
    17.334492, 16.255480, 24.932312, 31.667524, 11747.007813,
    18.771877, 26.783592, 10000000000.0
};
static const int dr[4] = {0, 1, 1, 1}, dc[4] = {1, 0, 1, -1};

typedef struct { int next[3], fail; double score; } DFA;
typedef struct { uint64_t key; double score; int depth, flag, action; } Entry;
typedef struct { int action; double rank; } Move;
typedef struct {
    int n, total, filled, width, budget, nodes, stopped;
    unsigned char board[CELLS];
    int lines[LINES][MAX_N], lengths[LINES], line_of[CELLS][4], count;
    double line_score[LINES], score;
    uint64_t keys[CELLS][3], hash;
    Entry table[TT_N];
    const DFA *dfa;
} Search;

static int inside(int r, int c, int n) { return r >= 0 && c >= 0 && r < n && c < n; }
static uint64_t random64(uint64_t *x) {
    uint64_t z = (*x += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static void build_dfa(DFA *d) {
    memset(d, 0, sizeof(DFA) * DFA_N);
    int count = 1;
    for (int side = 0; side < 2; ++side) for (int p = 0; p < 16; ++p) {
        int len = (int)strlen(patterns[p]), palindrome = 1;
        for (int k = 0; k < len; ++k) if (patterns[p][k] != patterns[p][len-1-k]) palindrome = 0;
        for (int reverse = 0; reverse < (palindrome ? 1 : 2); ++reverse) {
            int node = 0;
            for (int k = 0; k < len; ++k) {
                int symbol = patterns[p][reverse ? len-1-k : k] - '0';
                if (side && symbol) symbol = 3 - symbol;
                if (!d[node].next[symbol]) d[node].next[symbol] = count++;
                node = d[node].next[symbol];
            }
            d[node].score += weights[p] * (side ? -2.079848 : 2.0);
        }
    }
    int queue[DFA_N], head = 0, tail = 0;
    for (int x = 0; x < 3; ++x) if (d[0].next[x]) queue[tail++] = d[0].next[x];
    while (head < tail) {
        int node = queue[head++];
        d[node].score += d[d[node].fail].score;
        for (int x = 0; x < 3; ++x) {
            int child = d[node].next[x];
            if (child) {
                d[child].fail = d[d[node].fail].next[x];
                queue[tail++] = child;
            } else d[node].next[x] = d[d[node].fail].next[x];
        }
    }
}
static double score_line(const Search *s, int line) {
    double result = 0; int node = 0;
    for (int i = 0; i < s->lengths[line]; ++i) {
        node = s->dfa[node].next[s->board[s->lines[line][i]]];
        result += s->dfa[node].score;
    }
    return result;
}
static void initialize(Search *s, const int8_t *board, int n, int player, const DFA *dfa) {
    s->n = n; s->total = n*n; s->dfa = dfa;
    uint64_t seed = 1234567;
    for (int a = 0; a < n*n; ++a) {
        s->keys[a][1] = random64(&seed); s->keys[a][2] = random64(&seed);
        if (board[a]) {
            s->board[a] = board[a] == player ? 1 : 2;
            s->hash ^= s->keys[a][s->board[a]];
            s->filled++;
        }
    }
    for (int dir = 0; dir < 4; ++dir) for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) {
        if (inside(r-dr[dir], c-dc[dir], n)) continue;
        int line = s->count++;
        for (int y=r, x=c; inside(y,x,n); y+=dr[dir], x+=dc[dir]) {
            int a=y*n+x;
            s->lines[line][s->lengths[line]++] = a;
            s->line_of[a][dir] = line;
        }
        s->line_score[line] = score_line(s, line);
        s->score += s->line_score[line];
    }
}
static void place(Search *s, int a, int player) {
    int previous = s->board[a];
    if (previous) { s->hash ^= s->keys[a][previous]; s->filled--; }
    s->board[a] = player;
    if (player) { s->hash ^= s->keys[a][player]; s->filled++; }
    for (int d=0; d<4; ++d) {
        int line=s->line_of[a][d];
        s->score -= s->line_score[line];
        s->line_score[line] = score_line(s,line);
        s->score += s->line_score[line];
    }
}
static int wins(const Search *s, int a, int player) {
    for (int d=0; d<4; ++d) {
        int count=1, r=a/s->n, c=a%s->n;
        for (int sign=-1; sign<=1; sign+=2) {
            int y=r+sign*dr[d], x=c+sign*dc[d];
            while (inside(y,x,s->n) && s->board[y*s->n+x]==player) {
                count++; y+=sign*dr[d]; x+=sign*dc[d];
            }
        }
        if (count>=5) return 1;
    }
    return 0;
}
static int compare_moves(const void *aa, const void *bb) {
    const Move *a=aa, *b=bb;
    if (a->rank != b->rank) return a->rank < b->rank ? 1 : -1;
    return a->action - b->action;
}
static int candidates(Search *s, int player, Move *moves) {
    if (!s->filled) { moves[0]=(Move){(s->n/2)*s->n+s->n/2, 0}; return 1; }
    unsigned char near[CELLS]={0};
    for (int a=0; a<s->total; ++a) if (s->board[a]) {
        for (int d=0; d<4; ++d) for (int k=-2; k<=2; ++k) if (k) {
            int r=a/s->n+k*dr[d], c=a%s->n+k*dc[d];
            if (inside(r,c,s->n)) near[r*s->n+c]=1;
        }
    }
    int count=0, blocks=0;
    for (int a=0; a<s->total; ++a) if (near[a] && !s->board[a]) {
        if (wins(s,a,player)) { moves[0]=(Move){a, INF}; return 1; }
        if (wins(s,a,3-player)) blocks++;
        moves[count++]=(Move){a, 0};
    }
    if (blocks) {
        int kept=0;
        for (int i=0; i<count; ++i) if (wins(s,moves[i].action,3-player)) moves[kept++]=moves[i];
        return kept;
    }
    double original=s->score;
    for (int i=0; i<count; ++i) {
        place(s,moves[i].action,player);
        moves[i].rank = player==1 ? s->score : -s->score;
        place(s,moves[i].action,0);
        s->score=original; /* avoid accumulated floating point undo drift */
    }
    qsort(moves,count,sizeof(Move),compare_moves);
    return s->width && count>s->width ? s->width : count;
}
static double search(Search *s, int player, int depth, int ply, int last, double alpha, double beta, int *choice) {
    if (s->budget && s->nodes >= s->budget) { s->stopped=1; return 0; }
    s->nodes++;
    if (last>=0 && wins(s,last,3-player)) return player==2 ? WIN-ply : -WIN+ply;
    if (s->filled==s->total) return 0;
    if (!depth) return s->score;
    uint64_t key=s->hash ^ (player==1 ? UINT64_C(0xa123b456c789) : UINT64_C(0xb987a654d321));
    Entry *entry=&s->table[key % TT_N];
    double original_alpha=alpha, original_beta=beta;
    if (entry->flag && entry->key==key && entry->depth>=depth) {
        if (entry->flag==1) { *choice=entry->action; return entry->score; }
        if (entry->flag==2 && entry->score>alpha) alpha=entry->score;
        if (entry->flag==3 && entry->score<beta) beta=entry->score;
        if (alpha>=beta) { *choice=entry->action; return entry->score; }
    }
    Move moves[CELLS]; int count=candidates(s,player,moves), best_action=-1;
    if (!count) return 0;
    double best=player==1 ? -INF : INF;
    for (int i=0; i<count; ++i) {
        int a=moves[i].action, ignored=-1; double old=s->score;
        place(s,a,player);
        double value=search(s,3-player,depth-1,ply+1,a,alpha,beta,&ignored);
        place(s,a,0); s->score=old;
        if (s->stopped) return 0;
        if (best_action<0 || (player==1 ? value>best : value<best)) { best=value; best_action=a; }
        if (player==1 && best>alpha) alpha=best;
        if (player==2 && best<beta) beta=best;
        if (alpha>=beta) break;
    }
    *choice=best_action;
    *entry=(Entry){key,best,depth,best<=original_alpha ? 3 : best>=original_beta ? 2 : 1,best_action};
    return best;
}

/* Returns -1 for terminal boards. Output stats: nodes, fully completed depth. */
void teacher_batch(const int8_t *boards, const int *players, int batch, int n,
                   int depth, int width, int budget, int workers,
                   int *actions, double *scores, int *nodes, int *depths) {
    DFA dfa[DFA_N]; build_dfa(dfa);
    #pragma omp parallel for schedule(dynamic) num_threads(workers)
    for (int b=0; b<batch; ++b) {
        Search *s=calloc(1,sizeof(Search));
        actions[b]=-1; scores[b]=0; nodes[b]=0; depths[b]=0;
        if (!s) { actions[b]=-2; continue; }
        initialize(s,boards+b*n*n,n,players[b],dfa);
        s->width=width; s->budget=budget;
        int terminal=s->filled==s->total;
        for (int a=0; a<s->total && !terminal; ++a) if (s->board[a] && wins(s,a,s->board[a])) terminal=1;
        if (!terminal) {
            Move initial[CELLS]; int count=candidates(s,1,initial);
            if (count) actions[b]=initial[0].action;
            for (int d=1; d<=depth; ++d) {
                int action=-1;
                double value=search(s,1,d,0,-1,-INF,INF,&action);
                if (s->stopped) break;
                if (action>=0) { actions[b]=action; scores[b]=value; depths[b]=d; }
                if (fabs(value)>WIN/2) break;
            }
        }
        nodes[b]=s->nodes;
        free(s);
    }
}
/* Multi-PV labels: every root candidate gets a full-window search at the same
 * completed depth. Never train on fail-low bounds or a partly completed pass.
 * The node budget is shared across all candidates and iterative depths. */
void teacher_policy_batch(const int8_t *boards, const int *players, int batch, int n,
                          int depth, int width, int budget, int workers,
                          double *scores, int *nodes, int *depths) {
    DFA dfa[DFA_N]; build_dfa(dfa);
    #pragma omp parallel for schedule(dynamic) num_threads(workers)
    for (int b=0; b<batch; ++b) {
        double *out=scores+b*n*n;
        for (int a=0; a<n*n; ++a) out[a]=-INFINITY;
        nodes[b]=0; depths[b]=0;
        Search *s=calloc(1,sizeof(Search));
        if (!s) continue;
        initialize(s,boards+b*n*n,n,players[b],dfa);
        s->width=width; s->budget=budget;
        int terminal=s->filled==s->total;
        for (int a=0; a<s->total && !terminal; ++a)
            if (s->board[a] && wins(s,a,s->board[a])) terminal=1;
        if (!terminal) {
            Move moves[CELLS]; int count=0;
            /* Preserve ALL immediate wins, rather than the first found one. */
            for (int a=0; a<s->total; ++a)
                if (!s->board[a] && wins(s,a,1)) moves[count++]=(Move){a,0};
            if (count) {
                for (int i=0; i<count; ++i) out[moves[i].action]=WIN-1;
                depths[b]=1;
            } else {
                count=candidates(s,1,moves);
                double values[CELLS];
                /* Explicit depth-0 fallback if the budget cannot complete even
                 * one pass. Static values are computed for EVERY candidate. */
                for (int i=0; i<count; ++i) {
                    int a=moves[i].action; double old=s->score;
                    place(s,a,1);
                    out[a]=s->filled==s->total ? 0 : s->score;
                    place(s,a,0); s->score=old;
                }
                for (int d=1; d<=depth && count; ++d) {
                    for (int i=0; i<count; ++i) {
                        int a=moves[i].action, ignored=-1; double old=s->score;
                        place(s,a,1);
                        values[i]=search(s,2,d-1,1,a,-INF,INF,&ignored);
                        place(s,a,0); s->score=old;
                        if (s->stopped) break;
                    }
                    if (s->stopped) break;
                    double best=-INF;
                    for (int i=0; i<count; ++i) {
                        out[moves[i].action]=values[i];
                        if (values[i]>best) best=values[i];
                    }
                    depths[b]=d;
                    if (fabs(best)>WIN/2) break;
                }
            }
        }
        nodes[b]=s->nodes;
        free(s);
    }
}

/* Exposed for independent parity tests of the inherited pattern evaluation. */
double teacher_score(const int8_t *board, int n, int player) {
    DFA dfa[DFA_N]; build_dfa(dfa);
    Search *s=calloc(1,sizeof(Search));
    if (!s) return NAN;
    initialize(s,board,n,player,dfa);
    double value=s->score;
    free(s); return value;
}
