package Practical2;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;

public class WaterJugProblem {
    static class State {
        int x, y;

        public State(int x, int y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null || getClass() != obj.getClass())
                return false;
            State state = (State) obj;
            return x == state.x && y == state.y;
        }

        @Override
        public int hashCode() {
            return x * 31 + y;
        }
    }

    static class Node {
        State state;
        Node parent;

        public Node(State state, Node parent) {
            this.state = state;
            this.parent = parent;
        }
    }

    static void waterJugProblem(int jug1Capacity, int jug2Capacity, int target) {
        State initialState = new State(0, 0);

        Queue<Node> queue = new LinkedList<>();
        queue.add(new Node(initialState, null));

        HashSet<State> visited = new HashSet<>();

        while (!queue.isEmpty()) {
            Node currentNode = queue.poll();
            State currentState = currentNode.state;

            if (currentState.x == target || currentState.y == target) {
                printSolution(currentNode);
                return;
            }

            visited.add(currentState);

            // Fill jug 1
            State nextState = new State(jug1Capacity, currentState.y);
            enqueueIfNotVisited(new Node(nextState, currentNode), visited, queue);

            // Fill jug 2
            nextState = new State(currentState.x, jug2Capacity);
            enqueueIfNotVisited(new Node(nextState, currentNode), visited, queue);

            // Empty jug 1
            nextState = new State(0, currentState.y);
            enqueueIfNotVisited(new Node(nextState, currentNode), visited, queue);

            // Empty jug 2
            nextState = new State(currentState.x, 0);
            enqueueIfNotVisited(new Node(nextState, currentNode), visited, queue);

            // Pour water from jug 1 to jug 2
            int pourAmount = Math.min(currentState.x, jug2Capacity - currentState.y);
            nextState = new State(currentState.x - pourAmount, currentState.y + pourAmount);
            enqueueIfNotVisited(new Node(nextState, currentNode), visited, queue);

            // Pour water from jug 2 to jug 1
            pourAmount = Math.min(jug1Capacity - currentState.x, currentState.y);
            nextState = new State(currentState.x + pourAmount, currentState.y - pourAmount);
            enqueueIfNotVisited(new Node(nextState, currentNode), visited, queue);
        }

        System.out.println("No solution exists.");
    }

    static void enqueueIfNotVisited(Node node, HashSet<State> visited, Queue<Node> queue) {
        if (!visited.contains(node.state)) {
            queue.add(node);
            visited.add(node.state);
        }
    }

    static void printSolution(Node node) {
        LinkedList<Node> path = new LinkedList<>();
        while (node != null) {
            path.addFirst(node);
            node = node.parent;
        }

        for (Node n : path) {
            System.out.println("Jug 1: " + n.state.x + " Jug 2: " + n.state.y);
        }
    }

    public static void main(String[] args) {
        int jug1Capacity = 4;
        int jug2Capacity = 3;
        int target = 2;

        waterJugProblem(jug1Capacity, jug2Capacity, target);
    }
}
