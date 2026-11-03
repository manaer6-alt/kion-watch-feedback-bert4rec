"""Build a concise manuscript from audited numerical inputs and cited sources.

Also portable as repository/analysis/build_manuscript.py.
"""
import json
import re
import shutil
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = (ROOT/'reference_results').is_dir()
OUT = ROOT/'paper' if PORTABLE else ROOT/'studies/02_watch_feedback/article'
DATA = ROOT/'reference_results' if PORTABLE else ROOT/'outputs/watch_feedback/8f922f1b9331'
assert json.loads((DATA/'independent_review.json').read_text(encoding='utf-8'))['status']=='passed'
source = (OUT/'manuscript_source.md').read_text(encoding='utf-8')
sources = json.loads((OUT/'sources.json').read_text(encoding='utf-8'))
by_key = {row['key']:row for row in sources}
table = pd.read_csv(DATA/'independent_results_by_method.csv').set_index(['split','method'])
methods = ['ALL','DELETE10','TARGET10','STEP10','SOFT']


def num(x, places=5):
    return f'{x:.{places}f}'.replace('.',',')


def md_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(r)+' |' for r in rows])


source=source.replace('{{TABLE_METHODS}}',md_table(['Правило','История','Вес w(r)'],[
    ['ALL','Сохраняется','1'],
    ['DELETE10','Удаляются просмотры r ≤ 0,1','1 для оставшихся'],
    ['TARGET10','Сохраняется','0 при r ≤ 0,1; иначе 1'],
    ['STEP10','Сохраняется','1 при r ≤ 0,1; иначе 3'],
    ['SOFT','Сохраняется','0,2 + 0,8r']]))
source=source.replace('{{TABLE_PARAMETERS}}',md_table(['Параметр','Значение','Параметр','Значение'],[
    ['Transformer-блоки','2','Максимум на обучении','20 объектов'],
    ['Размерность','64','Вход при рекомендации','До 19 + MASK'],
    ['Головы внимания','2','Вероятность маски','0,35'],
    ['Внутренний слой','256','Dropout','0,1'],
    ['Оптимизатор','AdamW','Learning rate','0,001'],
    ['Weight decay','0,01','Размер мини-пакета','128'],
    ['Норма градиента','Не более 5','Точность','FP32'],
    ['Шаги на запуск','20 000','Валидация','Каждые 1000 шагов'],
    ['Seed','17, 29, 43','GPU','Tesla T4']]))
rows=[]
best={(split,metric):table.loc[split][metric+'_mean'].max() for split in ['main','replication']
      for metric in ['test_ndcg10_high','test_recall10_high']}
for method in methods:
    row=[method]
    for split in ['main','replication']:
        a=table.loc[(split,method)]
        value=num(a.test_ndcg10_high_mean)+' ± '+num(a.test_ndcg10_high_std)
        if a.test_ndcg10_high_mean==best[(split,'test_ndcg10_high')]: value='**'+value+'**'
        row.append(value)
    for split in ['main','replication']:
        a=table.loc[(split,method)]
        value=num(a.test_recall10_high_mean)
        if a.test_recall10_high_mean==best[(split,'test_recall10_high')]: value='**'+value+'**'
        row.append(value)
    rows.append(row)
source=source.replace('{{TABLE_RESULTS}}',md_table(['Метод','NDCG основной','NDCG повторный','Recall основной','Recall повторный'],rows))
loss_latex=r'\mathcal{L}=-\frac{\sum_{t\in M}w(r_t)\log P_\theta(i_t\mid\widetilde{s})}{\sum_{t\in M}w(r_t)}.'
source=source.replace('{{LOSS_EQUATION}}','$$\n'+loss_latex+'\n$$')
source=source.replace('{{OVERVIEW_FIGURE}}','![Качество рекомендаций, основные разности и время обучения](figures/figure_overview.png)')
posthoc=DATA/'posthoc'
if '{{POSTHOC_RESULTS}}' in source:
    assert json.loads((posthoc/'posthoc_validation.json').read_text(encoding='utf-8'))['status']=='passed'
    source=source.replace('{{POSTHOC_RESULTS}}',(posthoc/'article_fragment.md').read_text(encoding='utf-8').strip())
    source=source.replace('{{POSTHOC_METHODS}}',
        'Для дополнительного исследовательского сравнения SOFT и DELETE10 пользователи разделены по полной длине истории (10–19, 20–49, не менее 50 фильмов), доле просмотров ≤10% в последних максимум 19 объектах (0, до 1/3 включительно, более 1/3) и изменению длины контекста после удаления. Признаки определены до теста. Вклады фильмов сгруппированы по обучающей частоте (1–10, 11–100, более 100 зрителей) и рассчитаны с общим IDCG пользователя, поэтому суммируются в исходную NDCG. Этот анализ выполнен после получения общих результатов; его 95%-интервалы описательные и не учитывают множественность сравнений.')
if not PORTABLE:
    (OUT/'figures').mkdir(exist_ok=True)
    for suffix in ['png','svg']:
        shutil.copyfile(DATA/('figures_verified/figure_overview.'+suffix),OUT/('figures/figure_overview.'+suffix))
        if (posthoc/'figures'/('figure_posthoc.'+suffix)).exists():
            shutil.copyfile(posthoc/'figures'/('figure_posthoc.'+suffix),OUT/('figures/figure_posthoc.'+suffix))
order=[]


def cite(match):
    keys=[k.strip().lstrip('@') for k in match[1].split(';')]
    for k in keys:
        assert k in by_key,k
        if k not in order: order.append(k)
    return '['+', '.join(str(order.index(k)+1) for k in keys)+']'


source=re.sub(r'\[@([^\]]+)\]',cite,source)


def author_text(author):
    if ',' not in author: return author
    last, first=author.split(',',1)
    return last+' '+' '.join(part[0]+'.' for part in first.strip().replace('-',' ').split())


refs=[]
bib=[]
for i,k in enumerate(order,1):
    r=by_key[k]
    author=', '.join(author_text(a) for a in r['authors'][:3])+(' et al.' if len(r['authors'])>3 else '')
    year=(' '+str(r['year'])+'.') if r['year'] else ''
    link=('[DOI '+r['doi']+'](https://doi.org/'+r['doi']+')') if r['doi'] else '[URL]('+r['url']+')'
    refs.append(f"{i}. {author} {r['title']}. {r['publication']}.{year} {link}.")
    fields={'author':' and '.join(r['authors']),'title':r['title'],'howpublished':r['publication'],'url':r['url']}
    if r['year']: fields['year']=str(r['year'])
    if r['doi']: fields['doi']=r['doi']
    bib.append('@misc{'+k+',\n'+',\n'.join('  '+name+' = {'+value+'}' for name,value in fields.items())+'\n}')
source=source.replace('{{REFERENCES}}','\n\n'.join(refs))
assert '{{' not in source and '[@' not in source
(OUT/'manuscript_ru.md').write_text(source,encoding='utf-8',newline='\n')
(OUT/'references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8',newline='\n')
(OUT/'citation_map.json').write_text(json.dumps({k:i+1 for i,k in enumerate(order)},indent=2)+'\n',encoding='utf-8')

# One native editable Word equation; no raster equation screenshots.
def m(tag,text=None):
    node=OxmlElement('m:'+tag)
    if text is not None: node.text=text
    return node


def mr(text):
    node=m('r'); node.append(m('t',text)); return node


def sub(base,index):
    node=m('sSub'); e=m('e'); e.append(mr(base)); s=m('sub'); s.append(mr(index)); node.extend([e,s]); return node


def weighted_sum(body):
    n=m('nary'); pr=m('naryPr')
    char=m('chr'); char.set(qn('m:val'),'∑'); pr.append(char)
    hide=m('supHide'); hide.set(qn('m:val'),'1'); pr.append(hide)
    loc=m('limLoc'); loc.set(qn('m:val'),'undOvr'); pr.append(loc); n.append(pr)
    lower=m('sub'); lower.append(mr('t∈M')); n.append(lower); n.append(m('sup'))
    e=m('e'); e.extend(body); n.append(e); return n


def loss_equation():
    w=[mr('w('),sub('r','t'),mr(')')]
    numerator=m('num'); numerator.append(weighted_sum(w+[mr('log '),sub('P','θ'),mr('('),sub('i','t'),mr(' | s̃)')]))
    denominator=m('den'); denominator.append(weighted_sum([mr('w('),sub('r','t'),mr(')')]))
    fraction=m('f'); fraction.extend([numerator,denominator])
    math=m('oMath'); math.extend([mr('ℒ = −'),fraction])
    block=m('oMathPara'); block.append(math); return block


document=Document()
section=document.sections[0]
section.page_width,section.page_height=Cm(21),Cm(29.7)
section.top_margin=section.bottom_margin=Cm(1.8)
section.left_margin=section.right_margin=Cm(2)
for name in ['Normal','Title','Heading 1','Caption']:
    style=document.styles[name]
    style.font.name='Times New Roman'; style.font.color.rgb=RGBColor(0,0,0)
    fonts=style.element.get_or_add_rPr().rFonts
    for key in ['ascii','hAnsi','cs']: fonts.set(qn('w:'+key),'Times New Roman')
    for key in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']: fonts.attrib.pop(qn('w:'+key),None)
    for border in list(style.element.xpath('./w:pPr/w:pBdr')): border.getparent().remove(border)
document.styles['Normal'].font.size=Pt(11)
document.styles['Normal'].paragraph_format.line_spacing=1.06
document.styles['Normal'].paragraph_format.space_after=Pt(5)
document.styles['Title'].font.size=Pt(15)
document.styles['Title'].paragraph_format.space_after=Pt(9)
document.styles['Heading 1'].font.size=Pt(12)
document.styles['Heading 1'].paragraph_format.space_before=Pt(9)
document.styles['Heading 1'].paragraph_format.space_after=Pt(5)
document.styles['Heading 1'].paragraph_format.keep_with_next=True
document.styles['Caption'].font.size=Pt(10)
document.styles['Caption'].font.italic=False
document.styles['Caption'].paragraph_format.space_after=Pt(5)
document.core_properties.author=''
document.core_properties.last_modified_by=''
document.core_properties.title=source.splitlines()[0][2:]
document.core_properties.subject='Доля просмотра как неявная обратная связь в BERT4Rec на KION'


def formatted(p,text):
    # Actual hyperlinks with compact display labels keep references readable.
    for part in re.split(r'(\[[^\]]+\]\(https?://[^)]+\)|\*\*[^*]+\*\*)',text):
        link=re.fullmatch(r'\[([^\]]+)\]\((https?://[^)]+)\)',part)
        if link:
            from docx.opc.constants import RELATIONSHIP_TYPE
            relationship=p.part.relate_to(link[2],RELATIONSHIP_TYPE.HYPERLINK,is_external=True)
            node=OxmlElement('w:hyperlink'); node.set(qn('r:id'),relationship)
            r=OxmlElement('w:r'); t=OxmlElement('w:t'); t.text=link[1]; r.append(t); node.append(r); p._p.append(node)
        else:
            bold=part.startswith('**') and part.endswith('**')
            r=p.add_run(part[2:-2] if bold else part)
            if bold: r.bold=True


def add_table(lines):
    data=[[v.strip() for v in line.strip('|').split('|')] for line in lines]
    data=[data[0]]+data[2:]
    n=len(data[0]); widths={3:[2.1,7.0,7.9],4:[4.6,3.1,5.0,4.3],5:[2.0,4.3,4.3,3.2,3.2]}[n]
    t=document.add_table(rows=1,cols=n); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    for col,width in zip(t.columns,widths): col.width=Cm(width)
    for i,values in enumerate(data):
        if i: t.add_row()
        row=t.rows[i]; pr=row._tr.get_or_add_trPr(); pr.append(OxmlElement('w:cantSplit'))
        if not i: pr.append(OxmlElement('w:tblHeader'))
        for j,value in enumerate(values):
            cell=row.cells[j]; cell.width=Cm(widths[j]); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr(); borders=OxmlElement('w:tcBorders')
            for side in ['top','bottom','left','right']:
                border=OxmlElement('w:'+side)
                for k,v in [('val','single'),('sz','4'),('color','D9D9D9')]: border.set(qn('w:'+k),v)
                borders.append(border)
            pr.append(borders); margins=OxmlElement('w:tcMar')
            for side in ['top','bottom','left','right']:
                v=OxmlElement('w:'+side); v.set(qn('w:w'),'65' if side in ['top','bottom'] else '75'); v.set(qn('w:type'),'dxa'); margins.append(v)
            pr.append(margins); shade=OxmlElement('w:shd'); shade.set(qn('w:fill'),'E7ECF0' if not i else 'FFFFFF'); pr.append(shade)
            p=cell.paragraphs[0]; p.paragraph_format.space_before=Pt(1); p.paragraph_format.space_after=Pt(1)
            p.paragraph_format.line_spacing=1.02
            p.paragraph_format.keep_with_next=(i<len(data)-1)
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if (n==3 and j==1) or (n==4 and j in [0,2]) else WD_ALIGN_PARAGRAPH.CENTER
            formatted(p,value)
            for r in p.runs:
                r.font.size=Pt(9.5)
                if not i: r.bold=True
    p=document.add_paragraph(); p.paragraph_format.space_after=Pt(1); p.paragraph_format.line_spacing=.2


lines=source.splitlines(); i=0; references=False
while i<len(lines):
    line=lines[i].strip()
    if not line: i+=1; continue
    if line=='$$':
        i+=1
        while lines[i].strip()!='$$': i+=1
        p=document.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p._p.append(loss_equation())
        p.paragraph_format.keep_together=True
    elif line.startswith('|'):
        chunk=[]
        while i<len(lines) and lines[i].strip().startswith('|'): chunk.append(lines[i].strip()); i+=1
        add_table(chunk); continue
    elif line.startswith('!['):
        match=re.match(r'!\[([^]]+)\]\(([^)]+)\)',line)
        p=document.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next=True
        shape=p.add_run().add_picture(str(OUT/match[2]),width=Cm(17))
        shape._inline.docPr.set('descr',match[1])
    elif line.startswith('# '):
        p=document.add_paragraph(line[2:],style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    elif line.startswith('## '):
        title=line[3:]; document.add_paragraph(title,style='Heading 1'); references=title=='Литература'
    else:
        caption=bool(re.match(r'^(Таблица|Рисунок) \d+\.',line))
        p=document.add_paragraph(style='Caption' if caption else 'Normal'); formatted(p,line)
        if caption:
            p.paragraph_format.keep_with_next=line.startswith('Таблица'); p.paragraph_format.keep_together=True
        elif references:
            p.paragraph_format.line_spacing=1
            p.paragraph_format.space_after=Pt(4)
            for r in p.runs: r.font.size=Pt(9.5)
        else: p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    i+=1
target=OUT/'manuscript_ru.docx'; document.save(target)
report={'status':'authored_pending_visual_review','sources':len(order),'tables':len(document.tables),
        'figures':len(document.inline_shapes),'native_equations':len(document.element.xpath('//m:oMath')),
        'body_words':len(source.split('## Литература')[0].split()),'output':str(target)}
(OUT/'document_build.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
