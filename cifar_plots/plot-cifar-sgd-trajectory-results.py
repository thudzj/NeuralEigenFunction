import os
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

import pandas as pd
import numpy.random as rnd
import seaborn as sns

results = {'ResNet20':{'SGD':{'Negative Log-likelihood':0.3567, 'Accuracy':0.9194, 'Expected Calibration Error': 0.0486}, 
                       'SGD++':{'Negative Log-likelihood':0.2956, 'Accuracy':0.9196, 'Expected Calibration Error': 0.0291}, 
                       'SWA':{'Negative Log-likelihood':0.2390, 'Accuracy':0.9264, 'Expected Calibration Error': 0.0254}, 
                       'SWAG':{'Negative Log-likelihood':0.2247, 'Accuracy':0.9256, 'Expected Calibration Error': 0.0071}, 
                       'SWA++':{'Negative Log-likelihood':0.2229, 'Accuracy':0.9256, 'Expected Calibration Error': 0.0079}},
            'ResNet32':{'SGD':{'Negative Log-likelihood':0.3696, 'Accuracy':0.9203, 'Expected Calibration Error': 0.0517}, 
                       'SGD++':{'Negative Log-likelihood':0.3081, 'Accuracy':0.9214, 'Expected Calibration Error': 0.0358}, 
                       'SWA':{'Negative Log-likelihood':0.2349, 'Accuracy':0.9271, 'Expected Calibration Error': 0.0265}, 
                       'SWAG':{'Negative Log-likelihood':0.2126, 'Accuracy':0.9264, 'Expected Calibration Error': 0.0058}, 
                       'SWA++':{'Negative Log-likelihood':0.2160, 'Accuracy':0.9276, 'Expected Calibration Error': 0.0067}},
            'ResNet56':{'SGD':{'Negative Log-likelihood':0.3358, 'Accuracy':0.9243, 'Expected Calibration Error': 0.0497}, 
                       'SGD++':{'Negative Log-likelihood':0.2711, 'Accuracy':0.9241, 'Expected Calibration Error': 0.0333}, 
                       'SWA':{'Negative Log-likelihood':0.2104, 'Accuracy':0.9359, 'Expected Calibration Error': 0.0263}, 
                       'SWAG':{'Negative Log-likelihood':0.1887, 'Accuracy':0.9343, 'Expected Calibration Error': 0.0055}, 
                       'SWA++':{'Negative Log-likelihood':0.1905, 'Accuracy':0.9364, 'Expected Calibration Error': 0.0051}},
            'ResNet110':{'SGD':{'Negative Log-likelihood':0.3455, 'Accuracy':0.9285, 'Expected Calibration Error': 0.0463}, 
                        'SGD++':{'Negative Log-likelihood':0.2873, 'Accuracy':0.9288, 'Expected Calibration Error': 0.0320}, 
                        'SWA':{'Negative Log-likelihood':0.2031, 'Accuracy':0.9385, 'Expected Calibration Error': 0.0245}, 
                        'SWAG':{'Negative Log-likelihood':0.1882, 'Accuracy':0.9371, 'Expected Calibration Error': 0.0059}, 
                        'SWA++':{'Negative Log-likelihood':0.1851, 'Accuracy':0.9387, 'Expected Calibration Error': 0.0048}},}


models = ['ResNet20', 'ResNet32', 'ResNet56', 'ResNet110',]

for typ in ['Negative Log-likelihood', 'Accuracy', 'Expected Calibration Error']:
    data = []
    for model, item in results.items():
        for method, sub_item in item.items():
            data.append((models.index(model), method, sub_item[typ] * (100 if typ == 'Accuracy' else 1)))
    
    data = np.array(data)
    df = pd.DataFrame(data = data, columns = ['Model', 'method', 'value'])
    df.value = df.value.astype(float)
    print(df)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    sns_plot = sns.barplot(x='Model', y='value', hue='method', data=df)
    sns_plot.set(
        ylabel=typ
    )
    handles, labels = ax.get_legend_handles_labels()
    if typ == 'Accuracy':
      sns_plot.set(
          ylabel= 'Accuracy (%)'
      )
      ax.legend(handles=handles, labels=labels, ncol=2)
      for p in sns_plot.patches:
        sns_plot.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.074, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black', fontsize=7)
    else:
      ax.get_legend().remove()
    ax.set_xticklabels(models)
    ax.spines['bottom'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if typ == 'Accuracy':
      ax.set_ylim(91, 94.5)
    elif typ == 'Negative Log-likelihood':
      ax.set_ylim(0.15, None)
    ax.set_xlabel(' ')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='lightgray', linestyle='--')
    print('corruption_{}'.format(typ) +'.pdf')
    plt.savefig('comparison_{}'.format(typ) +'.pdf', format='pdf', dpi=1000, bbox_inches='tight')
