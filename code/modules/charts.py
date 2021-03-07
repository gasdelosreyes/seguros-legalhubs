import numpy as np
import matplotlib.pyplot as plt

def legalhub_color_palet():
    return ['#371852','#646565','#00AA83','#CE6828','#532B81','#E3E3E3','#42BB93','#DBA086','#C397C5','#B5D9CF','E9CFC2']

def percentage_plus_quantity(pct, allvals):
    absolute = round(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def manualPCT(vector,value):
    pct = (value * 100)/np.sum(vector)
    return "{:.1f}%\n({:d})".format(pct, value)

def createPie(data=list,labels=list,title='Pie Chart Basic',type='pie',save='piechart.png'):
    '''
        Create a chart pie. If data lenght surpass 5, then it became a donut pie chart
        Parametros:
            data = Vector with the data
            label = Vector with the name of each data value
            title (OPTIONAL) = Title of the pie chart
            type (OPTIONAL) = chart type {pie, donut} - Default pie
            save (OPTIONAL) = root and file name - Default root/piechart.png
    '''
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(aspect="equal"))
    if(type == 'donut' or len(data) > 5):
        wedges, texts, percentage = ax.pie(data, wedgeprops=dict(width=0.5),autopct='',textprops=dict(color="w"), startangle=-40, colors=legalhub_color_palet())
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i] + '\n' + manualPCT(data,data[i]) , xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw)
    elif(type == 'pie'):
        explode = []
        for i in range(0,len(data)):
            explode.append(0.005)
        wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: percentage_plus_quantity(pct, data),
                                        textprops=dict(color="w"),colors=legalhub_color_palet(),explode=explode)
        ax.legend(wedges, labels,
                title="labels",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=8, weight="bold")
    
    ax.set_title(label=title,loc='left')
    plt.savefig(save)

createPie(data=[375,75,250,300,20,8,78],labels=['flour','sugar','butter','berries','doritos','blabla','prueba'])