import os
import ast
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
    max = data.groupby('n_reinas')['epochs'].max()
    min = data.groupby('n_reinas')['epochs'].min()
    #std = data.groupby('n_reinas')['epochs'].std()
    #print(mean[8], std[8])
    #up = mean+std
    #down = mean-std
    #q1 = data.groupby('n_reinas')['epochs'].quantile(0.25)
    #q3 = data.groupby('n_reinas')['epochs'].quantile(0.75)
    
    end = search_final_unique_value(mean)
    start = mean.index[0]
    
    mean_log = np.log10(mean)[:end-start]
    max = np.log10(max)[:end-start]
    min = np.log10(min)[:end-start]
    #q1_log = np.log10(q1)[:end-start]
    #q3_log = np.log10(q3)[:end-start]
    #std_log = np.log10(std)[:end-start]
    #print(down)
    #up = np.log10(up)[:end-start]
    #down = np.log10(down)[:end-start]
    
    plt.figure(figsize=figsize)
    plt.title('Crecimiento logaritmico de épocas según número de reinas')
    plt.ylabel('Épocas')
    plt.xlabel('N')
    plt.plot(mean_log, marker='o')
    

    max_value = int(mean_log.max())+0.5
    plt.fill_between(np.arange(mean.index[0],end), max, min, alpha=0.2)
    values = [i for i in np.arange(2,max_value,0.5)]+[max[end-1]]
    plt.yticks(values, [int(round(10**i)) for i in values])
    
    plt.savefig(f'./images/{name_file}_epocas.png', dpi=200)

def evolution_collitions(data, name_file, figsize):
    mean = data.groupby('n_reinas')['best_cost'].mean()
    #q1 = data.groupby('n_reinas')['best_cost'].quantile(0.25)
    #q3 = data.groupby('n_reinas')['best_cost'].quantile(0.75)
    std = data.groupby('n_reinas')['best_cost'].std()
    up = mean+std
    down = mean-std

    #std = mean.st
    plt.figure(figsize=figsize)
    plt.plot(mean, label='media')
    #plt.plot(q1, label='Q1')
    #plt.plot(q3, label='Q3')
    plt.title('Número de colisiones según el número de reinas')
    plt.xlabel('N')
    plt.ylabel('Colisiones')
    plt.legend()

    start = mean.index[0]
    plt.fill_between(np.arange(start, start+len(mean)), up, down, alpha=0.2)
    
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

def create_figure_temperature(data, file_name, figsize):
    data_auto = pd.DataFrame(columns=['tmax', 'tmin', 'steps', 'updates'])
    schedules = data['schedule'].values
    for i, schedule in enumerate(schedules):
        data_auto.loc[i,:] = ast.literal_eval(schedule)

    data_auto['tmax'] = data_auto['tmax'].astype('int')
    data_auto['n_reinas'] = data['n_reinas'].astype('int')
    
    mean = data_auto.groupby('n_reinas')['tmax'].mean().astype('int')
    std = data_auto.groupby('n_reinas')['tmax'].std().astype('float')
    up = mean+std
    down = mean-std
    #q1 = data_auto.groupby('n_reinas')['tmax'].quantile(0.25).astype('int')
    #q3 = data_auto.groupby('n_reinas')['tmax'].quantile(0.75).astype('int')

    plt.figure(figsize=figsize)
    plt.title('Evolución Tmax según número de reinas')
    plt.xlabel('N')
    plt.ylabel('Tmax')
    plt.plot(mean, marker='o')

    start = mean.index[0]
    end = mean.index[-1]
    plt.fill_between(np.arange(start, end+1), up, down, alpha=0.2)

    plt.savefig(f'./images/{file_name}_tmax.png', dpi=200)


def main():
    dirs = os.listdir('./results')
    if not os.path.exists('./images/'):
        os.mkdir('./images/')

    for dir in dirs:
        name_dir = dir[:-4]
        data = pd.read_csv(f'./results/{dir}')
        
        evolucion_epochs(data, name_dir, (10,5))
        evolution_collitions(data, name_dir, (10,5))
        create_table(data, name_dir)
        if name_dir.__contains__('auto'):
            create_figure_temperature(data, name_dir, (10,5))

if __name__ == '__main__':
    main()
    #data = pd.read_csv('./results/experiments_auto.csv')
    #create_figure_temperature(data, 'experiments_auto_grande', (10,5))
    #create_figure_temperature(data, 'experiments_auto_grande', (10,5))
