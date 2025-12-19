#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <memory>
#include <algorithm>
#include <chrono>
#include <random>

using namespace std;

/* ===================== RANDOM ENGINE ===================== */
static mt19937 rng(random_device{}());

/* ===================== COLOUR ===================== */
enum class Colour { NONE, RED, BLUE };

Colour opposite(Colour c) {
    if (c == Colour::RED) return Colour::BLUE;
    if (c == Colour::BLUE) return Colour::RED;
    return Colour::NONE;
}

/* ===================== MOVE ===================== */
struct Move {
    int x, y;
    Move(int x = -1, int y = -1) : x(x), y(y) {}
    string str() const { return to_string(x) + "," + to_string(y); }
};

/* ===================== BOARD ===================== */
struct Tile { Colour colour = Colour::NONE; };

class Board {
public:
    int size;
    vector<vector<Tile>> tiles;
    Board(int s = 5) : size(s), tiles(s, vector<Tile>(s)) {}
    void set_tile_colour(int x, int y, Colour c) { tiles[x][y].colour = c; }
    bool has_ended(Colour) const { return false; } // TODO: implement Hex win detection
};

/* ===================== SIMULATION ===================== */
vector<Move> generate_actions(const Board& state) {
    vector<Move> moves;
    for (int i = 0; i < state.size; ++i)
        for (int j = 0; j < state.size; ++j)
            if (state.tiles[i][j].colour == Colour::NONE)
                moves.emplace_back(i, j);
    return moves;
}

double state_to_result(const Board& state, Colour colour) {
    if (state.has_ended(colour)) return 1.0;
    if (state.has_ended(opposite(colour))) return 0.0;
    return 0.5;
}

pair<Board, vector<Move>> simulate(const Board& state, Colour colour) {
    Board sim = state;
    vector<Move> moves = generate_actions(sim);
    shuffle(moves.begin(), moves.end(), rng);

    Colour current = colour;
    vector<Move> played;
    for (auto& mv : moves) {
        sim.set_tile_colour(mv.x, mv.y, current);
        current = opposite(current);
        played.push_back(mv);
    }
    return {sim, played};
}

/* ===================== NODE ===================== */
struct Node : public enable_shared_from_this<Node> {
    Board state;
    Move move;
    Colour colour;
    int turn;
    shared_ptr<Node> parent;
    vector<shared_ptr<Node>> children;
    bool expanded = false;
    int visits = 0;
    double result = 0.0;
    map<string, int> rave_visits;
    map<string, double> rave_result;

    Node(Board s, Move m, Colour c, int t, shared_ptr<Node> p = nullptr)
        : state(s), move(m), colour(c), turn(t), parent(p) {}

    bool has_children() const { return !children.empty(); }

    shared_ptr<Node> expand() {
        expanded = true;
        auto actions = generate_actions(state);
        if (actions.empty()) return nullptr;

        shuffle(actions.begin(), actions.end(), rng); // shuffle moves to break bias

        for (auto& mv : actions) {
            Board next = state;
            next.set_tile_colour(mv.x, mv.y, opposite(colour));
            children.push_back(make_shared<Node>(next, mv, opposite(colour), turn + 1, shared_from_this()));
        }

        return children[rng() % children.size()];
    }

    pair<double, vector<Move>> rollout() {
        auto sim = simulate(state, opposite(colour));
        double res = state_to_result(sim.first, colour);
        return {res, sim.second};
    }

    void backpropagate(double res, const vector<Move>& moves) {
        visits++;
        result += res;
        for (auto& mv : moves) {
            string k = mv.str();
            rave_visits[k]++;
            rave_result[k] += res;
        }
        if (parent) parent->backpropagate(1.0 - res, moves);
    }

    double reward() const { return visits ? result / visits : 0.0; }
};

/* ===================== POLICIES ===================== */
shared_ptr<Node> ucb1_policy(shared_ptr<Node> node, double c = 1.414) {
    double best_score = -1e9;
    shared_ptr<Node> best_node;
    for (auto& ch : node->children) {
        if (ch->visits == 0) return ch;
        double score = ch->reward() + c * sqrt(log(node->visits) / ch->visits);
        if (score > best_score) {
            best_score = score;
            best_node = ch;
        }
    }
    return best_node;
}

/* ===================== MCTS ===================== */
Move mcts(shared_ptr<Node> root, double time_limit = 2.0) {
    auto start = chrono::steady_clock::now();
    while (true) {
        double elapsed = chrono::duration<double>(
            chrono::steady_clock::now() - start).count();
        if (elapsed >= time_limit) break;

        auto node = root;
        while (node->has_children())
            node = ucb1_policy(node);
        if (!node->expanded)
            node = node->expand();

        auto [res, moves] = node->rollout();
        node->backpropagate(res, moves);
    }

    auto best = *max_element(
        root->children.begin(), root->children.end(),
        [](auto a, auto b){ return a->reward() < b->reward(); }
    );
    return best->move;
}

/* ===================== MAIN LOOP ===================== */
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    while (true) {
        string token;
        if (!(cin >> token)) break; // stdin closed
        if (token != "SIZE") break;

        int size; cin >> size;

        cin >> token; // COLOUR
        string colour_str; cin >> colour_str;
        Colour my_colour = (colour_str == "RED") ? Colour::RED : Colour::BLUE;

        cin >> token; // TURN
        int turn; cin >> turn;

        cin >> token; // OPP
        string opp; cin >> opp;
        Move opp_move;
        if (opp != "NONE") { opp_move.x = stoi(opp); cin >> opp_move.y; }

        cin >> token; // BOARD
        Board board(size);
        for (int i = 0; i < size; ++i) {
            string row; cin >> row;
            for (int j = 0; j < size; ++j) {
                if (row[j] == 'R') board.tiles[i][j].colour = Colour::RED;
                else if (row[j] == 'B') board.tiles[i][j].colour = Colour::BLUE;
            }
        }
        cin >> token; // END

        auto root = make_shared<Node>(board, opp_move, my_colour, turn);
        Move best = mcts(root, 3.0);

        cout << "MOVE " << best.x << " " << best.y << endl;
        cout.flush();
    }

    return 0;
}
