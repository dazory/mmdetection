import sys

def get_command(type, seed, category, config_name):
    folder_name = f"{config_name}" if seed == 0 else f"{config_name}_seed{seed}"
    if type == 'train':
        comm = f'python3 -u /ws/external/tools/train.py ' \
               f'--seed {seed} ' \
               f'--deterministic /ws/external/configs/_dshong/{category}/{config_name}.py ' \
               f'--work-dir /ws/data/dshong/_dshong/{category}/{folder_name}'
    elif type == 'test':
        comm = f"python3 -u /ws/external/tools/analysis_tools/test_robustness.py " \
               f"/ws/data/dshong/_dshong/{category}/{folder_name}/{config_name}.py " \
               f"/ws/data/dshong/_dshong/{category}/{folder_name}/epoch_2.pth " \
               f"--out /ws/data/dshong/_dshong/{category}/{folder_name}/epoch_2.pkl " \
               f"--corruptions benchmark --eval bbox --load-dataset corrupted " \
               f"| tee /ws/data/dshong/_dshong/{category}/{folder_name}/test_robustness_result_2epoch.txt"
    elif type == 'parse':
        comm = f"python3 /ws/external/tools/analysis_tools/parse_txt2dict.py " \
               f"/ws/data/dshong/_dshong/{category}/{folder_name}/test_robustness_result_2epoch.txt " \
               f"/ws/data/dshong/_dshong/{category}/{folder_name}/{config_name}.py " \
               f"| tee /ws/data/dshong/_dshong/{category}/{folder_name}/summary_result_2epoch.txt"
    elif type == 'all':
        comm1 = get_command('train', seed, category, config_name)
        comm2 = get_command('test', seed, category, config_name)
        comm3 = get_command('parse', seed, category, config_name)
        comm = f"{comm1} && {comm2} && {comm3}"
    else:
        raise NotImplementedError
    return comm


import os
def main(type, seed, category, config_name, output_path):
    comm = get_command(type, seed, category, config_name)
    print(comm)
    with open(output_path, 'a') as f:
        f.write(comm)
        f.write('\n')
        f.close()
    os.system(f'chmod 777 {output_path}')

if __name__ == '__main__':
    # e.g. `python3 -u /ws/external/tools/analysis_tools/command_generator.py all 0 scale_effect city_augmix_s1 command.sh`
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])