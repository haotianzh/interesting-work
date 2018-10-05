from data_reader import *


class DoubanReader(DataReader):
    def __init__(self, data_path):
        super(DoubanReader, self).__init__(data_path)
        self.load_data()

    def load_data(self):
        train_data_path = self.data_path + 'train.data.pkl'
        valid_data_path = self.data_path + 'valid.data.pkl'
        test_data_path = self.data_path + 'test.data.pkl'

        if os.path.exists(train_data_path):
            self._load(self.vocab_path, train_data_path, valid_data_path, test_data_path)
        else:
            self._build_vocab(self._read_text(self.train_path), self.vocab_path)
            self.train_data = self._proc(self._read_text(self.train_path), self.vocab)
            self.valid_data = self._proc(self._read_text(self.valid_path), self.vocab)
            self.test_data = self._proc(self._read_text(self.test_path), self.vocab)
            self._save(self.vocab_path, train_data_path, valid_data_path, test_data_path, False)

    def _build_vocab(self, text_data, vocab_path):
        lines = text_data.replace('\t', ' ').split(EOS_TOKEN)
        words = []
        for l in lines: 
            if l != '':
                for token in l.strip().split(' '):
                    if token != '':
                        words.append(token)
        super(DoubanReader, self)._build_vocab(words, vocab_path, 5)
        # threshold == 1: 123945
        # threshold == 2: 54331
        # threshold == 3: 37229
        # threshold == 4: 30019
        # threshold == 5: 25088

    def _proc(self, text_data, vocab):
        lines = text_data.split(EOS_TOKEN)
        px, py = [], []
        for l in lines:
            if len(l.split('\t')) == 2:
                pxi, pyi = self.proc_single(l, vocab)
                px.append(pxi)
                py.append(pyi)
        return px, py

    def proc_single(self, line, vocab):
        x, y =  line.split('\t')

        px, py = [], []
        for w in x.split(' '):
            if w != '':
                px.append(vocab.get(w, 1))
        for w in y.split(' '):
            if w != '':
                py.append(vocab.get(w, 1))
        """
        x = x.split(' ')
        y = y.split(' ')
        px = [vocab.get(w, 1) for w in x]
        py = [vocab.get(w, 1) for w in y]
        """
        return px, py



if __name__ == '__main__':
    dr = DoubanReader('../data/')
    dr.load_data()

