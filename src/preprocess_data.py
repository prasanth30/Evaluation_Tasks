import pandas as pd

def read_txt(path):
    with open(path,'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    text_pairs = []
    mx = 0 
    mn = 100
    x = []
    for _line in lines:
        arr = _line.split(':')
        if len(arr) == 1:
            continue
        text_pairs.append({'Event Type': arr[0], \
                          'Feynman Diagram': arr[1],   \
                           'Amplitude': arr[-2],        \
                           'Squared Amplitude': arr[-1] \
                          })
    return text_pairs

final_pairs = [read_txt(
            f'/kaggle/input/squared-amplitudes-test-data/SYMBA - Test Data/QED-2-to-2-diag-TreeLevel-{i}.txt')
            for i in range(10)]
final_pairs = [xx for x in final_pairs for xx in x]

# df = pd.DataFrame(final_pairs,columns=['Event Type','Feynman Diagram', 'Amplitude', 'Squared Amplitude'],dtype=['str','str','str','str'])
df = pd.DataFrame(final_pairs, columns=['Event Type', 'Feynman Diagram', 'Amplitude', 'Squared Amplitude'])
df = df.astype({'Event Type': 'string', 'Feynman Diagram': 'string', 'Amplitude': 'string', 'Squared Amplitude': 'string'})

from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

train_df.to_csv('train.csv')
val_df.to_csv('val.csv')
test_df.to_csv('test.csv')

