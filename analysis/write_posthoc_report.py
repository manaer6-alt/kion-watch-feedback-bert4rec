"""Describe exploratory aggregates and generate the compact manuscript fragment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir',type=Path,default=ROOT/'reference_results/posthoc')
args=parser.parse_args()
out=args.results_dir
validation=json.loads((out/'posthoc_validation.json').read_text(encoding='utf-8'))
assert validation['status']=='passed'
groups=pd.read_csv(out/'user_subgroups.csv')
films=pd.read_csv(out/'film_popularity_contributions.csv').set_index(['split','group_index'])
items=pd.read_csv(out/'film_contributions.csv')
recommendations=pd.read_csv(out/'recommendation_properties.csv')
overall=pd.read_csv(out/'overall_SOFT_DELETE10.csv').set_index('split')


def num(x,n=6): return f'{x:.{n}f}'.replace('.',',').replace('-','−')


def ci(row): return '['+num(row.ndcg_low)+'; '+num(row.ndcg_high)+']'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


labels={'main':'Основной','replication':'Повторный'}
sections=[]
for axis,title in [('history','Длина полной истории'),('weak_share','Доля слабых просмотров в последних 19 объектах'),
                   ('context','Изменение контекста при удалении')]:
    part=groups.loc[groups.axis.eq(axis)]
    rows=[[labels[r.split],r.group,int(r.users),num(100*r.share,2),num(r.ndcg_difference),ci(r),int(r.positive_seed_differences)]
          for r in part.itertuples()]
    sections.append('## '+title+'\n\n'+table(['Срез','Группа','Пользователи','Доля, %','SOFT − DELETE10','95%-интервал','Положительных seed из 3'],rows))
    for split in labels:
        g=part.loc[part.split.eq(split)]
        np.testing.assert_allclose((g.share*g.ndcg_difference).sum(),overall.loc[split,'ndcg_difference'],atol=1e-12)
history=groups.loc[groups.axis.eq('history')].set_index(['split','group_index'])
weak=groups.loc[groups.axis.eq('weak_share')].set_index(['split','group_index'])
wrath=items.loc[items.item_id.eq(9728)].set_index('split')
net_fraction=float(wrath.loc['main','ndcg_contribution_difference']/overall.loc['main','ndcg_difference'])
facts={'status':'exploratory_verified','history_20_49_difference':{s:float(history.loc[(s,1),'ndcg_difference']) for s in labels},
       'popular_future_target_percent':{s:float(100*films.loc[(s,2),'future_target_share']) for s in labels},
       'popular_ndcg_contribution_difference':{s:float(films.loc[(s,2),'ndcg_contribution_difference']) for s in labels},
       'item_9728_main_share_of_net_SOFT_DELETE10_difference_percent':100*net_fraction,
       'item_9728_difference':{s:float(wrath.loc[s,'ndcg_contribution_difference']) for s in labels},
       'weak_group_shares':{s:[float(weak.loc[(s,g),'share']) for g in range(3)] for s in labels},
       'composition':validation['composition_decomposition'],
       'limits':'posthoc_descriptive_not_routing_policy_not_causal_no_long_tail_gain_demonstrated'}
(out/'posthoc_article_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
fragment=f'''Дополнительный анализ не выявил устойчивого выбора между SOFT и DELETE10 по одной длине истории или доле слабых просмотров. Например, для историй 20–49 фильмов средняя разность SOFT − DELETE10 меняется с {num(history.loc[('main',1),'ndcg_difference'])} на основном срезе до {num(history.loc[('replication',1),'ndcg_difference'])} на повторном. Доли групп по слабым просмотрам при этом близки: без таких просмотров — {num(100*weak.loc[('main',0),'share'],2)} и {num(100*weak.loc[('replication',0),'share'],2)}%, с долей до 1/3 включительно — {num(100*weak.loc[('main',1),'share'],2)} и {num(100*weak.loc[('replication',1),'share'],2)}% (рисунок 2А). Поэтому по этим данным нельзя обосновать переносимое правило переключения обработки для зрителей.

Разложение по фильмам локализует разность в популярной части каталога. На объекты с более чем 100 зрителями в обучающей когорте приходится {num(100*films.loc[('main',2),'future_target_share'],2)} и {num(100*films.loc[('replication',2),'future_target_share'],2)}% будущих целевых взаимодействий. Их аддитивный вклад в SOFT − DELETE10 составляет {num(films.loc[('main',2),'ndcg_contribution_difference'])} и {num(films.loc[('replication',2),'ndcg_contribution_difference'])} (рисунок 2Б). Для менее популярных групп убедительного выигрыша SOFT не наблюдается. На основном срезе вклад фильма Wrath of Man равен {num(wrath.loc['main','ndcg_contribution_difference'])} — около {num(100*net_fraction,0)}% суммарной разности именно SOFT − DELETE10; на повторном его вклад отрицателен. Это описательная декомпозиция метрики, не причинное объяснение смены лидера. Улучшение средней оценки не доказывает улучшение рекомендаций редких фильмов.

![Исследовательский анализ различий SOFT и DELETE10 по зрителям и фильмам](figures/figure_posthoc.png)

Рисунок 2. Дополнительный анализ SOFT − DELETE10. А — разности NDCG@10 внутри групп по доле прошлых просмотров ≤10%; Б — аддитивные вклады групп фильмов по числу зрителей на обучении, с общим IDCG пользователя. Интервалы описательные 95%, условные на обученных моделях, без поправки на множественные сравнения.
'''
(out/'article_fragment.md').write_text(fragment,encoding='utf-8',newline='\n')
film_rows=[]
for (split,g),r in films.iterrows():
    film_rows.append([labels[split],r['group'],int(r.catalog_films),int(r.future_targets),num(100*r.future_target_share,2),
                      num(r.ndcg_contribution_difference),'['+num(r.low)+'; '+num(r.high)+']'])
props=recommendations.groupby(['split','method'])[['coverage_items','mean_log1p_training_frequency']].agg(['mean','std'])
property_rows=[[labels[s],m,num(r[('coverage_items','mean')],1)+' ± '+num(r[('coverage_items','std')],1),
                num(r[('mean_log1p_training_frequency','mean')],3)] for (s,m),r in props.iterrows()]
top=pd.read_csv(out/'top5_absolute_film_contributions.csv')
top_rows=[[labels[r.split],int(r.item_id),r.title_orig,int(r.future_high_target_users),num(r.ndcg_contribution_difference)] for r in top.itertuples()]
decomp=validation['composition_decomposition']
report=f'''# Дополнительный анализ зрителей и фильмов

Анализ выполнен по сохранённым рекомендациям и исходному KION без нового обучения. Схема групп — `exploratory_plan.json`, её хеш — {validation['design_hash']}. Она зафиксирована после изучения общих результатов, до расчёта этих групп. Все интервалы здесь — описательные 95%, без поправки на множественный поиск, условные на обученных моделях. Это не подтверждающая проверка нового правила.

## Что добавилось к выводу статьи

{fragment.split('![')[0]}
## Как определены группы

История — все известные каталогу фильмы с наблюдаемым процентом до начала теста, до удаления и обрезки. Доля слабых просмотров считается по последним максимум 19 фильмам этой истории; слабый просмотр — ≤10%. Нулевая доля не равнозначна отсутствию таких просмотров во всём прошлом. `weak_refilled_19` означает, что после удаления окно содержит 19 оставшихся фильмов; `weak_shorter_19` — меньше 19. Когда слабых просмотров в последнем окне нет, входы двух моделей совпадают, но их обучение различается.

Фильмы разбиты по числу пользователей в обучающей части выбранной когорты из 50 тысяч человек, до удаления и обрезки. Это не будущая популярность и не частота по всем зрителям платформы. На панели фильмов сохранён IDCG исходного множества целей пользователя; сумма вкладов равна обычной NDCG@10. Пользователи могут иметь цели нескольких групп. Вклады фильма или группы не являются результатами нового обучения после удаления этих объектов.

{chr(10).join(sections)}

Во всех трёх разбиениях доли групп почти одинаковы между срезами; направление эффекта внутри части групп меняется. Для длинных историй средние разности близки к нулю, но интервалы широки — отсутствие различий не доказано. Даже положительный интервал одной подгруппы не подтверждает универсальное превосходство: группы исследовательские, сравнений несколько, временной перенос ограничен.

## Аддитивные вклады групп фильмов

{table(['Срез','Частота','Фильмы каталога','Целевые пары','Доля целей, %','Вклад в разность','95%-интервал'],film_rows)}

Итоговая оценка качества в основном отражает популярные фильмы. Улучшение SOFT относительно ALL также сосредоточено в группе >100: это видно по разности столбцов SOFT_ndcg_contribution и ALL_ndcg_contribution в исходной CSV. Для редких групп мало попаданий, поэтому эти результаты не доказывают преимущество ни одного способа в длинном хвосте. Нельзя по числу пользователей в интервале судить о числе независимых наблюдений каждого редкого фильма.

## Отдельные фильмы

{table(['Срез','ID','Оригинальное название','Пользователи с этой целью','Вклад SOFT − DELETE10'],top_rows)}

Это пять наибольших по модулю наблюдаемых вкладов каждого среза. Названия не проходили отдельной проверки значимости. Доля {num(100*net_fraction,0)}% для Wrath of Man относится к суммарной разности SOFT − DELETE10 на основном срезе, с компенсацией положительных и отрицательных вкладов остальных фильмов. Она не означает 59% выигрыша SOFT относительно ALL и не описывает эффект исключения фильма из обучения. Пять крупнейших абсолютных вкладов составляют {num(100*validation['film_concentration']['main']['top5_absolute_share'],2)} и {num(100*validation['film_concentration']['replication']['top5_absolute_share'],2)}% суммы модулей вкладов соответственно; это другой знаменатель.

## Состав групп и смена среднего эффекта

Для разбиения по доле слабых просмотров симметричное арифметическое разложение SOFT − DELETE10 между основным и повторным срезами даёт:

- изменение общей разности: {num(decomp['main_minus_replication'])};
- слагаемое изменения долей групп: {num(decomp['composition'])};
- слагаемое изменения разностей внутри групп: {num(decomp['within_group'])}.

Сумма двух слагаемых точно восстанавливает изменение общей разности. В этом конкретном разбиении изменение долей групп мало; это не доказательство отсутствия любого сдвига аудитории или причинной роли конкретных фильмов. Срезы пересекаются: {validation['cohort_overlap']} пользователей общей исходной когорты, {validation['high_test_population_overlap']} пользователей обеих тестовых популяций высокой доли. Независимыми выборками они не объявляются.

## Свойства рекомендательных списков

{table(['Срез','Метод','Разные фильмы в рекомендациях, среднее ± SD','Средний log(1 + частота)'],property_rows)}

Покрытие считается по объединению top-10 всех пользователей высокой доли, отдельно для каждого seed. SOFT имеет меньшее среднее покрытие, чем DELETE10, на обоих срезах; это описательные средние трёх запусков. Относительная средняя популярность меняет направление: SOFT выше DELETE10 на основном срезе, ниже на повторном. Поэтому единственного объяснения вида «SOFT всегда сильнее сдвигает рекомендации к популярным» здесь нет.

## Проверка и файлы

По исходным данным восстановлены первые будущие дни и релевантные множества; их отпечатки совпадают с исходным тестом. Пользовательские NDCG и Recall заново рассчитаны по top-10 для ALL, DELETE10 и SOFT, всех трёх seed. Контекстные длины совпадают с сохранёнными. Каждое разбиение пользователей восстанавливает общую разность; вклады групп и отдельных фильмов восстанавливают NDCG. Пороги групп не подбирались по найденным разностям. Ни пользовательские строки, ни ID пользователей в файлы этого анализа не выгружаются.

Агрегаты: `user_subgroups.csv`, `film_popularity_contributions.csv`, `film_contributions.csv`, `recommendation_properties.csv`, `composition_decomposition.csv`. Машинная проверка — `posthoc_validation.json`; числа для статьи — `posthoc_article_facts.json`. Полный рисунок — `figures/figure_posthoc_full.png` и `.svg`; компактный рисунок статьи — `figures/figure_posthoc.png` и `.svg`.
'''
(out/'POSTHOC_ANALYSIS_RU.md').write_text(report,encoding='utf-8',newline='\n')
print(out/'POSTHOC_ANALYSIS_RU.md')
