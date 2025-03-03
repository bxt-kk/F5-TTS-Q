from glob import glob
import os
import random


class ESDDraw:

    def __init__(self, root_dir:str):

        label_paths = sorted(glob(os.path.join(root_dir, '*', '*.txt')))

        datas = []
        for path in label_paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    file_id, text, kind = line.split('\t')
                    kind = {
                        '中立': 'Neutral',
                        '生气': 'Angry',
                        '快乐': 'Happy',
                        '伤心': 'Sad',
                        '惊喜': 'Surprise',
                    }.get(kind, kind)
                    filename = os.path.join(file_id.split('_')[0], kind, file_id + '.wav')
                    datas.append((filename, text))
        self.datas = datas
        self.root_dir = root_dir

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, ix:int) -> tuple[str, str, str]:
        ref_filename, ref_text = self.datas[ix]
        for _ in range(10):
            gen_filename, gen_text = random.choice(self.datas)
            if gen_filename != ref_filename: break
        return os.path.join(self.root_dir, ref_filename), ref_text, gen_text

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]
