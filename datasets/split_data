import os
import numpy as np
import pandas as pd
import argparse


def create_dirs(base_dir):
    os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)


def save_split_data(data, file_path, columns):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Split Settings')
    parser.add_argument('--dataset', type=str, default="biosnap", choices=[ "bindingdb", "biosnap"])
    parser.add_argument('--split_settings', type=str, default="cluster",choices=["random", "cold_drug", "cold_target", "cold_pair", "cluster"])
    args = parser.parse_args()

    # Load full dataset
    base_path = os.path.join('./datasets/', args.dataset)
    full_data_path = os.path.join(base_path, 'fulldata.csv')
    df = pd.read_csv(full_data_path)

    split_dir = os.path.join(base_path, args.split_settings)
    create_dirs(split_dir)

    if args.split_settings == "cluster":
        # Use cluster IDs to split source and target
        source, target = [], []
        trg_cluster = np.array(list(set(df['target_cluster'])))
        drug_cluster = np.array(list(set(df['drug_cluster'])))
        np.random.shuffle(trg_cluster)
        np.random.shuffle(drug_cluster)

        trg_src, trg_tgt = np.split(trg_cluster, [int(0.6 * len(trg_cluster))])
        drug_src, drug_tgt = np.split(drug_cluster, [int(0.6 * len(drug_cluster))])

        smiledict, seqdict = {}, {}
        train_samples, valtest_samples = [], []
        for row in df.itertuples(index=False):
            smiles, sequence, interaction, d_cluster, t_cluster = row
            if smiles not in smiledict:
                smiledict[smiles] = len(smiledict)
            if sequence not in seqdict:
                seqdict[sequence] = len(seqdict)

            data_row = [smiles, sequence, interaction, d_cluster, t_cluster]
            smiles_idx = smiledict[smiles]
            seq_idx = seqdict[sequence]
            triple = [smiles_idx, seq_idx, int(interaction)]

            if d_cluster in drug_src and t_cluster in trg_src:
                source.append(data_row)
                train_samples.append(triple)
            elif d_cluster in drug_tgt and t_cluster in trg_tgt:
                target.append(data_row)
                valtest_samples.append(triple)

        # Split target into val/test
        target_train = target[:int(0.8 * len(target))]
        target_test = target[int(0.8 * len(target)):]
        val_samples = valtest_samples[:int(0.8 * len(valtest_samples))]
        test_samples = valtest_samples[int(0.8 * len(valtest_samples)):]

        # Save
        save_split_data(source, os.path.join(split_dir, 'source_train.csv'),
                        ['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster'])
        save_split_data(target_train, os.path.join(split_dir, 'target_train.csv'),
                        ['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster'])
        save_split_data(target_test, os.path.join(split_dir, 'target_test.csv'),
                        ['SMILES', 'Protein', 'Y', 'drug_cluster', 'target_cluster'])
        save_split_data(train_samples, os.path.join(split_dir, 'train/samples.csv'),
                        ['smiles', 'sequence', 'interactions'])
        save_split_data(val_samples, os.path.join(split_dir, 'valid/samples.csv'),
                        ['smiles', 'sequence', 'interactions'])
        save_split_data(test_samples, os.path.join(split_dir, 'test/samples.csv'),
                        ['smiles', 'sequence', 'interactions'])

    elif args.split_settings in ["cold_target", "cold_drug", "cold_pair"]:
        smiledict, seqdict = {}, {}
        smilelist, seqlist = [], []
        train, valtest = [], []
        train_samples, valtest_samples = [], []

        # Create index maps
        for row in df.itertuples(index=False):
            smiles, sequence, *_ = row
            if smiles not in smiledict:
                smiledict[smiles] = len(smiledict)
                smilelist.append(smiles)
            if sequence not in seqdict:
                seqdict[sequence] = len(seqdict)
                seqlist.append(sequence)

        smile_indices = np.array(list(smiledict.values()))
        seq_indices = np.array(list(seqdict.values()))
        np.random.shuffle(smile_indices)
        np.random.shuffle(seq_indices)

        if args.split_settings == "cold_target":
            train_idx, test_idx = np.split(seq_indices, [int(0.7 * len(seq_indices))])
            train_filter = lambda s, q: seqdict[q] in train_idx
            test_filter = lambda s, q: seqdict[q] in test_idx

        elif args.split_settings == "cold_drug":
            train_idx, test_idx = np.split(smile_indices, [int(0.7 * len(smile_indices))])
            train_filter = lambda s, q: smiledict[s] in train_idx
            test_filter = lambda s, q: smiledict[s] in test_idx

        elif args.split_settings == "cold_pair":
            drug_train, drug_test = np.split(smile_indices, [int(0.7 * len(smile_indices))])
            seq_train, seq_test = np.split(seq_indices, [int(0.7 * len(seq_indices))])
            train_filter = lambda s, q: smiledict[s] in drug_train and seqdict[q] in seq_train
            test_filter = lambda s, q: smiledict[s] in drug_test and seqdict[q] in seq_test

        for row in df.itertuples(index=False):
            smiles, sequence, interaction = row[:3]
            triple = [smiledict[smiles], seqdict[sequence], int(interaction)]
            if train_filter(smiles, sequence):
                train.append([smiles, sequence, interaction])
                train_samples.append(triple)
            elif test_filter(smiles, sequence):
                valtest.append([smiles, sequence, interaction])
                valtest_samples.append(triple)

        # Split validation/test
        val = valtest[:int(0.5 * len(valtest))]
        test = valtest[int(0.5 * len(valtest)):]
        val_samples = valtest_samples[:int(0.5 * len(valtest_samples))]
        test_samples = valtest_samples[int(0.5 * len(valtest_samples)):]

        # Save
        save_split_data(train, os.path.join(split_dir, 'train.csv'), ['SMILES', 'Protein', 'Y'])
        save_split_data(val, os.path.join(split_dir, 'val.csv'), ['SMILES', 'Protein', 'Y'])
        save_split_data(test, os.path.join(split_dir, 'test.csv'), ['SMILES', 'Protein', 'Y'])
        save_split_data(train_samples, os.path.join(split_dir, 'train/samples.csv'),
                        ['smiles', 'sequence', 'interactions'])
        save_split_data(val_samples, os.path.join(split_dir, 'valid/samples.csv'),
                        ['smiles', 'sequence', 'interactions'])
        save_split_data(test_samples, os.path.join(split_dir, 'test/samples.csv'),
                        ['smiles', 'sequence', 'interactions'])
    elif args.split_settings == 'random':
        fudfll = df.sample(frac=1, random_state=42).reset_index(drop=True)  

        smiledictnum2str = {}
        smiledictstr2num = {}
        sqddictnum2str = {}
        sqdictstr2num = {}
        smilelist = []
        sequencelist = []

        trainsamples = []
        valsamples = []
        testsamples = []

        for no, data in enumerate(df.values):
            smiles, sequence, interaction = data[:3]
            if smiledictstr2num.get(smiles) is None:
                smiledictstr2num[smiles] = len(smiledictstr2num)
                smiledictnum2str[str(len(smiledictnum2str))] = smiles
                smilelist.append(smiles)
            if sqdictstr2num.get(sequence) is None:
                sqdictstr2num[sequence] = len(sqdictstr2num)
                sqddictnum2str[str(len(sqddictnum2str))] = sequence
                sequencelist.append(sequence)

        total_len = len(df)
        train_end = int(0.7 * total_len)
        valid_end = int(0.85 * total_len)

        train_data = df.iloc[:train_end]
        valid_data = df.iloc[train_end:valid_end]
        test_data = df.iloc[valid_end:]

        for data in train_data.values:
            smiles, sequence, interaction = data[:3]
            smilesidx = smiledictstr2num[smiles]
            sequenceidx = sqdictstr2num[sequence]
            trainsamples.append([smilesidx, sequenceidx, int(interaction)])

        for data in valid_data.values:
            smiles, sequence, interaction = data[:3]
            smilesidx = smiledictstr2num[smiles]
            sequenceidx = sqdictstr2num[sequence]
            valsamples.append([smilesidx, sequenceidx, int(interaction)])

        for data in test_data.values:
            smiles, sequence, interaction = data[:3]
            smilesidx = smiledictstr2num[smiles]
            sequenceidx = sqdictstr2num[sequence]
            testsamples.append([smilesidx, sequenceidx, int(interaction)])

        column = ['SMILES', 'Protein', 'Y']
        pd.DataFrame(columns=column, data=train_data).to_csv(split_dir + '/train.csv', index=False)
        pd.DataFrame(columns=column, data=valid_data).to_csv(split_dir + '/val.csv', index=False)
        pd.DataFrame(columns=column, data=test_data).to_csv(split_dir + '/test.csv', index=False)

        column = ['smiles', 'sequence', 'interactions']
        pd.DataFrame(columns=column, data=trainsamples).to_csv(split_dir + '/train/samples.csv', index=False)
        pd.DataFrame(columns=column, data=valsamples).to_csv(split_dir + '/valid/samples.csv', index=False)
        pd.DataFrame(columns=column, data=testsamples).to_csv(split_dir + '/test/samples.csv', index=False)

        print(f"Train: {len(trainsamples)}, Valid: {len(valsamples)}, Test: {len(testsamples)}")
