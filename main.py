import numpy as np

import interface


def main():
    try:
        # Task 1
        p_max1 = 1
        if input("\nSolve task #1? (+/-): ") == '+':
            probs, p_max1 = interface.task1()
            # Task 1.1 (additional)
            if input("Do you want to know how many flight complexes are needed for success payload "
                     "delivering? (+/-): ") == '+':
                n_fc, p_req = interface.get_n_fc(p_max1)
                print(f"Answer: {n_fc} flight complex(es) with single probability "
                      f"{round(p_max1, 5)}\ncan delivery payload to target with required probability {p_req}")

        # Task 2
        if input("\nSolve task #2? (+/-): ") == '+':
            r_max, p_max2 = interface.task2()
            # Task 2.1 (additional)
            full_prob = p_max1 * p_max2
            print(f"\nFull probability for best flight complex: P = {full_prob}")
            with open('results/results.txt', 'a') as file:
                file.write(f"\nFull probability for best flight complex: P = {full_prob}")
                if input("Do you want to know how many such flight complexes are needed for success "
                         "target defeat? (+/-): ") == '+':
                    n_fc, p_req = interface.get_n_fc(full_prob)
                    file.write(f"\n{n_fc} flight complex(es) with single probability "
                               f"{round(full_prob, 5)} can defeat target with required probability {p_req}")
                    print(f"Answer: {n_fc} flight complex(es) with single probability "
                          f"{round(full_prob, 5)}\ncan defeat target with required probability {p_req}")

        # Task 3
        if input("\nSolve task #3? (+/-): ") == '+':
            interface.task3()
    except ValueError:
        print(f"Main Error {ValueError}!")
        exit(-1)
    else:
        print("All done...")
    return 0


if __name__ == '__main__':
    main()
else:
    print("Fatal Error: no entering point!")
    exit(-1)
