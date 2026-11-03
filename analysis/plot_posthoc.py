"""Coherent exploratory subgroup figures from aggregate CSVs only."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir',type=Path,default=ROOT/'reference_results/posthoc')
parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/posthoc_figures')
args=parser.parse_args()
assert json.loads((args.results_dir/'posthoc_validation.json').read_text(encoding='utf-8'))['status']=='passed'
subgroups=pd.read_csv(args.results_dir/'user_subgroups.csv')
films=pd.read_csv(args.results_dir/'film_popularity_contributions.csv')
colors={'main':'#245B8A','replication':'#B46B30'}
markers={'main':'o','replication':'s'}
names={'main':'Основной срез','replication':'Повторный срез'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path',
                     'savefig.facecolor':'white','axes.edgecolor':'#5B6268','axes.linewidth':.7})


def panel(ax,axis,labels,title,film=False):
    frame=films if film else subgroups.loc[subgroups.axis.eq(axis)]
    for split,offset in [('main',.13),('replication',-.13)]:
        part=frame.loc[frame.split.eq(split)].sort_values('group_index')
        for y,(_,row) in zip([2,1,0],part.iterrows()):
            value=row.ndcg_contribution_difference if film else row.ndcg_difference
            low=row.low if film else row.ndcg_low
            high=row.high if film else row.ndcg_high
            ax.errorbar(value,y+offset,xerr=np.array([[value-low],[high-value]]),fmt=markers[split],
                        color=colors[split],capsize=2.5,markersize=4.5,lw=1.05)
    ax.axvline(0,ls='--',lw=.8,color='#6C7378')
    ax.set(yticks=[2,1,0],yticklabels=labels,ylim=(-.5,2.5),
           xlabel='Вклад в разность NDCG@10' if film else 'Разность NDCG@10')
    ax.set_title(title,loc='left',pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(4,steps=[1,2,3,4,5,6,8,10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:.3f}'.replace('.',',')))
    ax.grid(axis='x',color='#E1E5E8',lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)


handles=[Line2D([0],[0],marker=markers[s],color=colors[s],lw=1,label=names[s]) for s in colors]
args.output_dir.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(6.7,3.05))
panel(axes[0],'weak_share',['0','Более 0 до 1/3','Более 1/3'],'А  Доля слабых просмотров')
panel(axes[1],None,['1–10','11–100','Более 100'],'Б  Частота фильма на обучении',film=True)
fig.subplots_adjust(left=.15,right=.98,top=.84,bottom=.30,wspace=.60)
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
for suffix in ['png','svg']:
    fig.savefig(args.output_dir/('figure_posthoc.'+suffix),dpi=360,bbox_inches='tight',pad_inches=.12)
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(8,5.9))
panel(axes[0,0],'history',['10–19','20–49','Не менее 50'],'А  Полная история до теста')
panel(axes[0,1],'weak_share',['0','Более 0 до 1/3','Более 1/3'],'Б  Доля слабых просмотров')
panel(axes[1,0],'context',['Нет слабых','Окно восполнено','Окно укорочено'],'В  Контекст после удаления')
panel(axes[1,1],None,['1–10','11–100','Более 100'],'Г  Частота фильма на обучении',film=True)
fig.subplots_adjust(left=.17,right=.98,top=.92,bottom=.17,hspace=.67,wspace=.62)
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False)
for suffix in ['png','svg']:
    fig.savefig(args.output_dir/('figure_posthoc_full.'+suffix),dpi=360,bbox_inches='tight',pad_inches=.12)
plt.close(fig)
(args.output_dir/'CAPTION_RU.md').write_text('''Дополнительный исследовательский анализ SOFT − DELETE10.
Интервалы — описательные пользовательские 95%, условные на обученных моделях, без поправки на множественные сравнения.
На пользовательских панелях показана разность средних NDCG внутри группы. Признаки истории доступны до теста.
На панели фильмов — аддитивные вклады групп в общую разность, со старым IDCG по всем целям пользователя.
Эти вклады суммируются в общую разность; они не являются отдельной перенормированной NDCG группы фильмов.
Частота фильма — число пользователей выбранной когорты в обучающей части до фильтрации и обрезки.
''',encoding='utf-8',newline='\n')
print(args.output_dir)
