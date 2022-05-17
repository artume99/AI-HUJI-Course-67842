import itertools
import sys


def proposition_string_template(propositions) -> str:
    propositions_str = " ".join(propositions)
    return f'Propositions:\n{propositions_str}\n'


def action_string_template(name, pre, add, delete) -> str:
    pre_str = " ".join(pre)
    add_str = " ".join(add)
    delete_str = " ".join(delete)
    return f'Name: {name}\n' \
           f'pre: {pre_str}\n' \
           f'add: {add_str}\n' \
           f'delete: {delete_str}\n'


def create_domain_file(domain_file_name, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    disks_pegs = [f'{disk}-{peg}' for disk, peg in itertools.product(disks, pegs)]
    free_disks = ['u' + disk for disk in disks]
    free_pegs = ['u' + peg for peg in pegs]
    disks_on_disks = [f'{disk1}ON{disk2}' for disk1, disk2 in itertools.combinations(disks, r=2)]
    domain_file = open(domain_file_name, 'w')  # use domain_file.write(str) to write to domain_file
    "*** YOUR CODE HERE ***"
    domain_file.write(proposition_string_template(disks_pegs + disks_on_disks + free_disks + free_pegs))
    domain_file.write("Actions:\n")
    for disk_peg, peg in itertools.product(disks_pegs, pegs):
        disk, curr_peg = disk_peg.split("-")
        if curr_peg == peg:
            continue
        name = disk_peg + peg
        pre = ["u" + disk, "u" + peg, disk_peg]
        add = [f'{disk}-{peg}']
        delete = [disk_peg, "u" + peg]
        domain_file.write(action_string_template(name, pre, add, delete))
    domain_file.close()


def initial_state_template(initial_state):
    initial_state_str = " ".join(initial_state)
    return f'Initial state: {initial_state_str}\n'

def goal_state_template(goal_state):
    goal_state_str = " ".join(goal_state)
    return f'Goal state: {goal_state_str}'


def write_peg_state(disks, pegs, peg_index):
    disks_on_disks = [f'{disk1}-{disk2}' for disk1, disk2 in itertools.combinations(disks, r=2)]
    final_state = []
    for dod in disks_on_disks:
        d1, d2 = dod.split("-")
        if int(d1[2]) == int(d2[2]) - 1:
            final_state.append(dod)

    peg = pegs[peg_index]
    for disk in disks:
        final_state.append(f'{disk}-{peg}')

    return final_state


def create_problem_file(problem_file_name_, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    problem_file = open(problem_file_name_, 'w')  # use problem_file.write(str) to write to problem_file
    "*** YOUR CODE HERE ***"
    initial_state = write_peg_state(disks, pegs, 0)
    goal_state = write_peg_state(disks, pegs, m - 1)
    problem_file.write(initial_state_template(initial_state))
    problem_file.write(goal_state_template(goal_state))

    problem_file.close()


if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print('Usage: hanoi.py n m')
    #     sys.exit(2)

    # n = int(float(sys.argv[1]))  # number of disks
    # m = int(float(sys.argv[2]))  # number of pegs

    n = 3
    m = 3

    domain_file_name = 'hanoi_%s_%s_domain.txt' % (n, m)
    problem_file_name = 'hanoi_%s_%s_problem.txt' % (n, m)

    create_domain_file(domain_file_name, n, m)
    create_problem_file(problem_file_name, n, m)
