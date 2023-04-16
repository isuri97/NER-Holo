import csv

# import pandas as pd
# #
# # with open('data/finalf_usm-test.csv', 'r') as in_file:
# #     stripped = (line.strip() for line in in_file)
# #     lines = (line.split(",") for line in stripped if line)
# #     with open('nn.csv', 'w') as out_file:
# #         writer = csv.writer(out_file, delimiter='\t')
# #         writer.writerow(('sentence_id', 'words', 'labels'))
# #         writer.writerows(lines)
# # file = 'data/finalf_usm-test.csv'
# # df = pd.read_csv(file)
# #
with open('nn-yale.csv', 'a') as out_file:
    with open('data/finalf_yale.txt', 'r') as in_file:
        lines = in_file.readlines()
        for line in lines:
            if len(line.split(',')) == 3:
                out_file.write(line.replace(',','\t'))
#
#

# import pandas as pd
# # #
# # # # Read the tab-separated values (TSV) file into a DataFrame
# df = pd.read_csv('dataset78.csv', header=None, sep='\t',quoting=csv.QUOTE_NONE, encoding='utf-8')
# df.head(10)
# # #




