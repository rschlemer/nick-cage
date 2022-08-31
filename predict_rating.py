import pickle
import pandas as pd
import argparse
from statistics import mean


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('duration', help='duration of movie in seconds')
    ap.add_argument('IMDB', help='audience rating on IMDB')
    ap.add_argument(
        'rotten',
        help='critics rating on rotten tomatos out of 10',
    )
    ap.add_argument('year', help='year film released')

    args = vars(ap.parse_args())

    data = pd.DataFrame(
        [
            [
                int(args['duration']),
                float(args['IMDB']),
                float(args['rotten']),
                int(args['year']) - 1982,
            ]
        ],
        columns=[
            'Duration',
            'IMDB',
            'Rotten',
            'Year',
        ],
    )

    scores = []

    print('')

    for p in ['tyler', 'hotdog', 'chris', 'ryan']:

        with open(f'output/{p}.pickle', 'rb') as f:
            clf = pickle.load(f)

        scores.append(round(clf.predict(data)[0], 2))
        print(f'predicted score for {p} is {scores[-1]}')

    print(f'average predicted score is {round(mean(scores),2)}')
    print('')
