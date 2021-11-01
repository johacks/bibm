import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

def search_final_unique_value(data):
    last_value = data.unique()[-1]
    first_index = data.index[0]
    for i in range(first_index, len(data)+first_index):
        if data[i] == last_value:
            return i-1
        
def evolucion_epochs(data, name_file, figsize):
    
    mean = data.groupby('n_reinas')['epochs'].mean()
    q1 = data.groupby('n_reinas')['epochs'].quantile(0.25)
    q3 = data.groupby('n_reinas')['epochs'].quantile(0.75)
    
    end = search_final_unique_value(mean)
    start = mean.index[0]
    
    mean_log = np.log10(mean)[:end-start]
    q1_log = np.log10(q1)[:end-start]
    q3_log = np.log10(q3)[:end-start]
    
    plt.figure(figsize=figsize)
    plt.title('Crecimiento logaritmico de épocas según número de reinas')
    plt.ylabel('Épocas')
    plt.xlabel('N')
    plt.plot(mean_log, marker='o')
    
    max_value = int(mean_log.max())+0.5
    plt.fill_between(np.arange(mean.index[0],end), q3_log, q1_log, alpha=0.2)
    values = [i for i in np.arange(2,max_value,0.5)]+[q3_log[end-1]]
    plt.yticks(values, [int(round(10**i)) for i in values])
    
    plt.savefig(f'./images/{name_file}_epocas.png', dpi=200)

def evolution_collitions(data, name_file, figsize):
    mean = data.groupby('n_reinas')['best_cost'].mean()
    q1 = data.groupby('n_reinas')['best_cost'].quantile(0.25)
    q3 = data.groupby('n_reinas')['best_cost'].quantile(0.75)

    plt.figure(figsize=figsize)
    plt.plot(mean, label='media')
    plt.plot(q1, label='Q1')
    plt.plot(q3, label='Q3')
    plt.title('Número de colisiones según el número de reinas')
    plt.xlabel('N')
    plt.ylabel('Colisiones')
    plt.legend()

    start = mean.index[0]
    plt.fill_between(np.arange(start, start+len(mean)), q3, q1, alpha=0.2)
    
    plt.savefig(f'./images/{name_file}_colisiones.png', dpi=200)

def create_table(data, file_name):
    df= pd.DataFrame()
    df['mean'] = data.groupby('n_reinas')['epochs'].mean().astype('int')
    df['max'] = data.groupby('n_reinas')['epochs'].max().astype('int')
    df['min'] = data.groupby('n_reinas')['epochs'].min().astype('int')

    df['Q1'] = data.groupby('n_reinas')['epochs'].quantile(0.25).astype('int')
    df['Q3'] = data.groupby('n_reinas')['epochs'].quantile(0.75).astype('int')
    df = df.reset_index()

    start = df.index[0]
    end = search_final_unique_value(df['mean'])
    fig =  ff.create_table(df[:end-start+2])
    fig.update_layout(
        autosize=True,
        width=500,
        height=350,
    )
    fig.show()
    fig.write_image(f'./images/{file_name}_table.png', scale=2)

def main():
    dirs = os.listdir('./results')
    if not os.path.exists('./images/'):
        os.mkdir('./images/')

    for dir in dirs:
        name_dir = dir[:-4]
        data = pd.read_csv(f'./results/{dir}')
        evolution_collitions(data, name_dir, (10,5))
        evolucion_epochs(data, name_dir, (10,5))
        create_table(data, name_dir)

if __name__ == '__main__':
    main()
