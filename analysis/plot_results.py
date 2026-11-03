"""One coherent overview built only from independently verified study-2 results."""
import json
import argparse
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(ROOT/'.work-tmp/matplotlib')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir',type=Path,default=ROOT/'reference_results')
parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/figures')
args=parser.parse_args()
out=args.results_dir
def available(checked,original):
    candidate=out/checked
    return candidate if candidate.exists() else out/original
table=pd.read_csv(available('independent_results_by_method.csv','results_by_method.csv')).set_index(['split','method'])
runs=pd.read_csv(available('independent_results_per_run.csv','results_per_run.csv'))
intervals=json.loads((out/'paired_intervals.json').read_text(encoding='utf-8'))
assert len(table)==10 and len(runs)==30 and len([r for r in intervals if r['role']=='primary'])==4
methods=['ALL','DELETE10','TARGET10','STEP10','SOFT']
colors=['#245B8A','#84919C','#B46B30','#7060A2','#267151']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10.2,
                     'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path',
                     'savefig.facecolor':'white','axes.edgecolor':'#5B6268','axes.linewidth':.7})
fig,axes=plt.subplots(2,2,figsize=(8.0,5.6),layout='constrained')

def polish(ax,decimals=3):
    # Integer mantissas avoid rounding a 0.0025 tick to a misleading 0.003 label.
    ax.xaxis.set_major_locator(MaxNLocator(4,steps=[1,2,3,4,5,6,8,10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{x:.{decimals}f}'.replace('.',',')))
    ax.grid(axis='x',color='#E1E5E8',lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)

for ax,split,title in zip(axes[0],['main','replication'],['А  Основной срез','Б  Повторение во времени']):
    for y,method,color in zip([4,3,2,1,0],methods,colors):
        r=table.loc[(split,method)]
        ax.errorbar(r.test_ndcg10_high_mean,y,xerr=r.test_ndcg10_high_std,
                    marker='o',color=color,capsize=3,markersize=4.8,lw=1.1)
    ax.set(yticks=[4,3,2,1,0],yticklabels=methods,ylim=(-.6,4.6),xlabel='NDCG@10: доля просмотра ≥50%')
    ax.set_title(title,loc='left',pad=9)
    polish(ax)

ax=axes[1,0]
primary=[r for r in intervals if r['role']=='primary']
labels=[]
for y,row in zip([3,2,1,0],primary):
    ax.hlines(y,row['low'],row['high'],color='#267151',lw=1.2)
    ax.vlines([row['low'],row['high']],y-.07,y+.07,color='#267151',lw=1.1)
    ax.plot(row['mean_difference'],y,'o',color='#267151',markersize=4.8)
    labels.append(('Осн.' if row['split']=='main' else 'Повт.')+': '+row['left']+' − '+row['right'])
ax.axvline(0,color='#657078',lw=.9,ls='--')
ax.set(yticks=[3,2,1,0],yticklabels=labels,ylim=(-.55,3.55),xlabel='Разность NDCG@10')
ax.set_title('В  Основные разности и интервалы',loc='left',pad=9)
polish(ax)
ax.tick_params(axis='y',labelsize=7.9)

ax=axes[1,1]
times=runs.groupby('method').training_seconds.agg(['mean','std'])
for y,method,color in zip([4,3,2,1,0],methods,colors):
    ax.errorbar(times.loc[method,'mean'],y,xerr=times.loc[method,'std'],marker='o',
                color=color,capsize=3,markersize=4.8,lw=1.1)
ax.set(yticks=[4,3,2,1,0],yticklabels=methods,ylim=(-.6,4.6),xlabel='Обучение, с (6 запусков на метод)')
ax.set_title('Г  Вычислительные затраты',loc='left',pad=9)
polish(ax,0)
fig.get_layout_engine().set(h_pad=.13,w_pad=.06,hspace=.08,wspace=.06)
dest=args.output_dir
dest.mkdir(parents=True,exist_ok=True)
for suffix in ['png','svg']:
    fig.savefig(dest/('figure_overview.'+suffix),dpi=360,bbox_inches='tight',pad_inches=.12)
plt.close(fig)
(dest/'README_RU.md').write_text('''Один рисунок из предоставленных таблиц результатов.
А/Б: среднее ± SD трёх seed каждого временного среза, NDCG@10 высокой доли.
В: четыре основных абсолютных разности с условными пользовательскими интервалами 98,75%.
Г: среднее ± SD времени по шести запускам метода (два среза × три seed), без валидации и теста.
''',encoding='utf-8')
print(dest)
