import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logs_name', type=str, required=True)
args = parser.parse_args()

path = './'
# patten = re.compile(r'log_entlm_crf_(\w+)_.*.txt')
# patten = re.compile(r'log_entlm_crf_.*_(.*?).txt')
# patten = re.compile(r'log_(\w+)_entlm_crf_.*.txt')
# patten = re.compile(r'training_multi_(.*)_template_log.txt')
logs_name = args.logs_name
results = ""
head = 'protocol'.ljust(10) + 'strict'.ljust(20) + 'exact'.ljust(20) + 'file_name'.ljust(30)+ '\n'
results += head

flist = os.listdir(path)
for f in flist:

    if f.startswith(logs_name):
        strict = ""
        exact = ""
        protocol = f.split(logs_name)[1][1:-4]
        with open(os.path.join(path,f), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                if line.startswith('strict'):
                    strict = line.split()[1]
                if line.startswith('exact'):
                    exact = line.split()[1]
        if strict and exact:
            results += protocol.ljust(10) + strict.ljust(20) + exact.ljust(20) + f.ljust(30) +'\n'
print(results)
result_file = './result_' + logs_name +'.txt'

with open(result_file, 'w', encoding='utf-8') as fw:
    fw.write(results)





