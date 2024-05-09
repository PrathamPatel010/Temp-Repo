package Practical3;

public class CryptarithmeticSolver {

    static void solveCryptarithmetic() {
        for (int S = 1; S <= 9; S++) {
            for (int E = 0; E <= 9; E++) {
                for (int N = 0; N <= 9; N++) {
                    for (int D = 0; D <= 9; D++) {
                        for (int M = 1; M <= 9; M++) {
                            for (int O = 0; O <= 9; O++) {
                                for (int R = 0; R <= 9; R++) {
                                    for (int Y = 0; Y <= 9; Y++) {
                                        int send = S * 1000 + E * 100 + N * 10 + D;
                                        int more = M * 1000 + O * 100 + R * 10 + E;
                                        int money = M * 10000 + O * 1000 + N * 100 + E * 10 + Y;

                                        if (send + more == money) {
                                            System.out.println(" S E N D ");
                                            System.out.println(" " + S + " " + E + " " + N + " " + D);
                                            System.out.println(" M O R E ");
                                            System.out.println(" " + M + " " + O + " " + R + " " + E);
                                            System.out.println(" --------- ");
                                            System.out.println(" " + M + " " + O + " " + N + " " + E + " " + Y);
                                            return; // Stop searching after finding one solution
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        System.out.println("No solution found.");
    }

    public static void main(String[] args) {
        solveCryptarithmetic();
    }
}
