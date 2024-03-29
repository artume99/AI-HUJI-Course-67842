import itertools
import sys


def proposition_string_template(propositions) -> str:
    propositions_str = " ".join(propositions)
    return propositions_str + '\n'


def action_string_template(name, pre, add, delete) -> str:
    pre_str = " ".join(pre)
    add_str = " ".join(add)
    delete_str = " ".join(delete)
    return f'Name: {name}\n' \
           f'pre: {pre_str}\n' \
           f'add: {add_str}\n' \
           f'delete: {delete_str}\n'


def move_action(file, disk, src, dest):
    name = f'MOVE_{disk}_FROM_{src}_TO_{dest}'
    pre = ['u' + disk, 'u' + dest, f'{disk}-{src}']
    add = [f'{disk}-{dest}', f'u{src}']
    delete = ['u' + dest, f'{disk}-{src}', ]
    file.write(action_string_template(name, pre, add, delete))


def write_actions(file, disks: list, pegs: list):
    file.write("Actions:\n")
    # move disk from disk to disk
    for disk, disk1, disk2 in itertools.combinations(disks, r=3):
        move_action(file, disk, disk1, disk2)
        move_action(file, disk, disk2, disk1)
    # move disk from peg to peg
    for disk in disks:
        for peg1, peg2 in itertools.combinations(pegs, r=2):
            move_action(file, disk, peg1, peg2)
            move_action(file, disk, peg2, peg1)
    # move disk from peg to disk (and vice versa)
    for m_disk in disks:
        for disk, peg in itertools.product(disks, pegs):
            n1 = int(m_disk.split("_")[1])
            n2 = int(disk.split("_")[1])
            if n1 >= n2:
                continue
            move_action(file, m_disk, peg, disk)
            move_action(file, m_disk, disk, peg)


def write_propositions(file, disks, pegs):
    file.write(f'Propositions:\n')
    disks_on_pegs = [f'{disk}-{peg}' for disk, peg in itertools.product(disks, pegs)]
    disks_on_disks = [f'{disk1}-{disk2}' for disk1, disk2 in itertools.combinations(disks, r=2)]
    free_disks = ['u' + disk for disk in disks]
    free_pegs = ['u' + peg for peg in pegs]
    file.write(proposition_string_template(disks_on_pegs + disks_on_disks + free_disks + free_pegs))


def create_domain_file(domain_file_name, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    domain_file = open(domain_file_name, 'w')  # use domain_file.write(str) to write to domain_file
    "*** YOUR CODE HERE ***"
    write_propositions(domain_file, disks, pegs)
    write_actions(domain_file, disks, pegs)
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
        n1 = int(d1.split("_")[1])
        n2 = int(d2.split("_")[1])
        if n1 == n2 - 1:
            final_state.append(dod)

    peg = pegs[peg_index]
    disk = disks[-1]
    final_state.append(f'{disk}-{peg}')

    return final_state


def add_empty_pegs(pegs):
    if len(pegs) == 1:
        return []
    free_pegs = ['u' + peg for peg in pegs[1:]]
    return free_pegs


def create_problem_file(problem_file_name_, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    problem_file = open(problem_file_name_, 'w')  # use problem_file.write(str) to write to problem_file
    "*** YOUR CODE HERE ***"
    initial_state = write_peg_state(disks, pegs, 0)
    initial_state += add_empty_pegs(pegs)
    initial_state.append("ud_0")
    goal_state = write_peg_state(disks, pegs, m_ - 1)
    problem_file.write(initial_state_template(initial_state))
    problem_file.write(goal_state_template(goal_state))

    problem_file.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: hanoi.py n m')
        sys.exit(2)

    n = int(float(sys.argv[1]))  # number of disks
    m = int(float(sys.argv[2]))  # number of pegs

    domain_file_name = 'hanoi_%s_%s_domain.txt' % (n, m)
    problem_file_name = 'hanoi_%s_%s_problem.txt' % (n, m)

    create_domain_file(domain_file_name, n, m)
    create_problem_file(problem_file_name, n, m)
