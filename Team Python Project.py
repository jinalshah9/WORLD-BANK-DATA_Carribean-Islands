#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:45:33 2019

@author: Team #4
"""


#Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def flag_missing(df):
    """ Defining a flag function for flagging missing data"""
    for col in df:
        if(df[col].isnull().astype(int).sum() > 0):
            df['m_'+col] = df[col].isnull().astype(int)
    return df



def impute_median(df):
    """Defining a function to impute with the median"""
    for col in df:
        if df[col].dtype != 'object':
            fill = df[col].median()
            df[col]=df[col].fillna(fill)
    return df


def prune(df, thresh = 9):
    """Defining a function to "prune" the dataset"""
    df_copy = pd.DataFrame.copy(df)
    for col in df_copy:
        if df_copy[col].isnull().sum() > thresh:
            del df_copy[col]
    return df_copy


def GDPcap(df):
    """ Defining a function to compute GDP/capita"""    
    CYM_rur = 0
    VGB_GDP = 1028000000
    
    df.loc[36,'Rural population (% of total population)'] = CYM_rur
    df.loc[27,'GDP (current US$)'] = VGB_GDP
    
    GDPpcapATG = df.loc[[6],'GDP (current US$)']/df.loc[[6],'Population, total']
    GDPpcapABW = df.loc[[9],'GDP (current US$)']/df.loc[[9],'Population, total']
    GDPpcapBHS = df.loc[[13],'GDP (current US$)']/df.loc[[13],'Population, total']
    GDPpcapBRB = df.loc[[16],'GDP (current US$)']/df.loc[[16],'Population, total']
    GDPpcapVGB = df.loc[[27],'GDP (current US$)']/df.loc[[27],'Population, total']
    GDPpcapCYM = df.loc[[36],'GDP (current US$)']/df.loc[[36],'Population, total']
    GDPpcapCUB = df.loc[[48],'GDP (current US$)']/df.loc[[48],'Population, total']
    GDPpcapDMA = df.loc[[54],'GDP (current US$)']/df.loc[[54],'Population, total']
    GDPpcapDOM = df.loc[[55],'GDP (current US$)']/df.loc[[55],'Population, total']
    GDPpcapGRD = df.loc[[78],'GDP (current US$)']/df.loc[[78],'Population, total']
    GDPpcapHTI = df.loc[[84],'GDP (current US$)']/df.loc[[84],'Population, total']
    GDPpcapJAM = df.loc[[97],'GDP (current US$)']/df.loc[[97],'Population, total']
    GDPpcapPRI = df.loc[[158],'GDP (current US$)']/df.loc[[158],'Population, total']
    GDPpcapKNA = df.loc[[163],'GDP (current US$)']/df.loc[[163],'Population, total']
    GDPpcapLCA = df.loc[[164],'GDP (current US$)']/df.loc[[164],'Population, total']
    GDPpcapVCT = df.loc[[165],'GDP (current US$)']/df.loc[[165],'Population, total']
    GDPpcapTTO = df.loc[[195],'GDP (current US$)']/df.loc[[195],'Population, total']
    GDPpcapTCA = df.loc[[199],'GDP (current US$)']/df.loc[[199],'Population, total']
    GDPpcapVIR = df.loc[[211],'GDP (current US$)']/df.loc[[211],'Population, total']
    
    GDPpcap_lst =[GDPpcapATG.iloc[0], GDPpcapABW.iloc[0], GDPpcapBHS.iloc[0], GDPpcapBRB.iloc[0], GDPpcapVGB.iloc[0], 
                  GDPpcapCYM.iloc[0], GDPpcapCUB.iloc[0], GDPpcapDMA.iloc[0], GDPpcapDOM.iloc[0], GDPpcapGRD.iloc[0],
                  GDPpcapHTI.iloc[0], GDPpcapJAM.iloc[0], GDPpcapPRI.iloc[0], GDPpcapKNA.iloc[0], GDPpcapLCA.iloc[0],
                  GDPpcapVCT.iloc[0], GDPpcapTTO.iloc[0], GDPpcapTCA.iloc[0], GDPpcapVIR.iloc[0]]
    df['GDP per capita'] = GDPpcap_lst
    
    
    
    return df
    


def data_input(df):
    """ Defining a function to input some data we found"""
    
    VGB_fem =0.5228*df.loc[27,'Population, total']
    CYM_fem =0.5129*df.loc[36,'Population, total']
    DMA_fem =0.4950*df.loc[54,'Population, total']
    KNA_fem =0.4996*df.loc[163,'Population, total']
    TCA_fem =0.4961*df.loc[199,'Population, total']
    VGB_mal =(1-0.5228)*df.loc[27,'Population, total']
    CYM_mal =(1-0.5129)*df.loc[36,'Population, total']
    DMA_mal =(1-0.4950)*df.loc[54,'Population, total']
    KNA_mal =(1-0.4996)*df.loc[163,'Population, total']
    TCA_mal =(1-0.4961)*df.loc[199,'Population, total']
    VGB_br =11.1
    DMA_br =15
    KNA_br =13
    TCA_br =14.9
    VGB_dr =5.2
    DMA_dr =7.9
    KNA_dr =7.2
    TCA_dr =3.3
    VGB_014 =16.72
    CYM_014 =17.91
    DMA_014 =21.62
    KNA_014 =20.09
    TCA_014 =21.62
    VGB_1564 =73.96
    CYM_1564 =68.96
    DMA_1564 =66.95
    KNA_1564 =70.88
    TCA_1564 =73.62
    VGB_65up =9.32
    CYM_65up =13.13
    DMA_65up =11.43
    KNA_65up =9.03
    TCA_65up =4.76
    VGB_life =78.9
    CYM_life =81.4
    DMA_life =77.4
    KNA_life =76.2
    TCA_life =80.1
    VGB_fert =1.3
    CYM_fert =1.84
    DMA_fert =2.03
    KNA_fert =1.77
    TCA_fert =1.7

    br_lst = [VGB_br, DMA_br, KNA_br, TCA_br]
    dr_lst = [VGB_dr, DMA_dr, KNA_dr, TCA_dr]
    fem_lst = [VGB_fem, CYM_fem, DMA_fem, KNA_fem, TCA_fem]
    mal_lst = [VGB_mal, CYM_mal, DMA_mal, KNA_mal, TCA_mal]
    p014_lst = [VGB_014, CYM_014, DMA_014, KNA_014, TCA_014]
    p1564_lst = [VGB_1564, CYM_1564, DMA_1564, KNA_1564, TCA_1564]
    p65up_lst = [VGB_65up, CYM_65up, DMA_65up, KNA_65up, TCA_65up]
    life_lst = [VGB_life, CYM_life, DMA_life, KNA_life, TCA_life]
    fert_lst = [VGB_fert, CYM_fert, DMA_fert, KNA_fert, TCA_fert]

    
    df.loc[[27,54,163,199],'Birth rate, crude (per 1,000 people)'] = br_lst
    df.loc[[27,54,163,199],'Death rate, crude (per 1,000 people)'] = dr_lst
    df.loc[[27,36,54,163,199],'Population, female'] = fem_lst
    df.loc[[27,36,54,163,199],'Population, male'] = mal_lst
    df.loc[[27,36,54,163,199],'Population ages 0-14 (% of total population)'] = p014_lst
    df.loc[[27,36,54,163,199],'Population ages 15-64 (% of total population)'] = p1564_lst
    df.loc[[27,36,54,163,199],'Population ages 65 and above (% of total population)'] = p65up_lst
    df.loc[[27,36,54,163,199],'Life expectancy at birth, total (years)'] = life_lst
    df.loc[[27,36,54,163,199],'Fertility rate, total (births per woman)'] = fert_lst

    return df


def flag_outliers(df):
    """ Defining a function to flag outliers"""


    df['out_Access to electricity (% of population)'] = 0
    df['out_Access to electricity (% of population)'].replace(to_replace = df.loc[[84], 'out_Access to electricity (% of population)'], value = 1, inplace = True)

    df['out_Access to electricity, rural (% of rural population)'] = 0
    df['out_Access to electricity, rural (% of rural population)'].replace(to_replace = df.loc[[84], 'out_Access to electricity, rural (% of rural population)'], value = 1, inplace = True)

    df['out_Access to electricity, urban (% of urban population)'] = 0
    df['out_Access to electricity, urban (% of urban population)'].replace(to_replace = df.loc[[84], 'out_Access to electricity, urban (% of urban population)'], value = 1, inplace = True)

    df['out_Adolescent fertility rate (births per 1,000 women ages 15-19)'] = 0
    df['out_Adolescent fertility rate (births per 1,000 women ages 15-19)'].replace(to_replace = df.loc[[55], 'out_Adolescent fertility rate (births per 1,000 women ages 15-19)'], value = 1, inplace = True)

    df['out_Age dependency ratio (% of working-age population)'] = 0
    df['out_Age dependency ratio (% of working-age population)'].replace(to_replace = df.loc[[84,211], 'out_Age dependency ratio (% of working-age population)'], value = 1, inplace = True)

    df['out_Age dependency ratio, old (% of working-age population)'] = 0
    df['out_Age dependency ratio, old (% of working-age population)'].replace(to_replace = df.loc[[16,158,211], 'out_Age dependency ratio, old (% of working-age population)'], value = 1, inplace = True)

    df['out_Age dependency ratio, young (% of working-age population)'] = 0
    df['out_Age dependency ratio, young (% of working-age population)'].replace(to_replace = df.loc[[55,84], 'out_Age dependency ratio, young (% of working-age population)'], value = 1, inplace = True)

    df['out_Agriculture, forestry, and fishing, value added (% of GDP)'] = 0
    df['out_Agriculture, forestry, and fishing, value added (% of GDP)'].replace(to_replace = df.loc[[54], 'out_Agriculture, forestry, and fishing, value added (% of GDP)'], value = 1, inplace = True)

    df['out_Birth rate, crude (per 1,000 people)'] = 0
    df['out_Birth rate, crude (per 1,000 people)'].replace(to_replace = df.loc[[55,84], 'out_Birth rate, crude (per 1,000 people)'], value = 1, inplace = True)

    df['out_Death rate, crude (per 1,000 people)'] = 0
    df['out_Death rate, crude (per 1,000 people)'].replace(to_replace = df.loc[[36], 'out_Death rate, crude (per 1,000 people)'], value = 1, inplace = True)

    df['out_Employment in agriculture (% of total employment) (modeled ILO estimate)'] = 0
    df['out_Employment in agriculture (% of total employment) (modeled ILO estimate)'].replace(to_replace = df.loc[[84], 'out_Employment in agriculture (% of total employment) (modeled ILO estimate)'], value = 1, inplace = True)

    df['out_Employment in services (% of total employment) (modeled ILO estimate)'] = 0
    df['out_Employment in services (% of total employment) (modeled ILO estimate)'].replace(to_replace = df.loc[[84], 'out_Employment in services (% of total employment) (modeled ILO estimate)'], value = 1, inplace = True)

    df['out_Fertility rate, total (births per woman)'] = 0
    df['out_Fertility rate, total (births per woman)'].replace(to_replace = df.loc[[55,84,158], 'out_Fertility rate, total (births per woman)'], value = 1, inplace = True)

    df['out_GDP (current US$)'] = 0
    df['out_GDP (current US$)'].replace(to_replace = df.loc[[48,55,158], 'out_GDP (current US$)'], value = 1, inplace = True)

    df['out_GDP growth (annual %)'] = 0
    df['out_GDP growth (annual %)'].replace(to_replace = df.loc[[54,158], 'out_GDP growth (annual %)'], value = 1, inplace = True)

    df['out_Industry (including construction), value added (% of GDP)'] = 0
    df['out_Industry (including construction), value added (% of GDP)'].replace(to_replace = df.loc[[158,195], 'out_Industry (including construction), value added (% of GDP)'], value = 1, inplace = True)

    df['out_Life expectancy at birth, total (years)'] = 0
    df['out_Life expectancy at birth, total (years)'].replace(to_replace = df.loc[[84], 'out_Life expectancy at birth, total (years)'], value = 1, inplace = True)

    df['out_Merchandise trade (% of GDP)'] = 0
    df['out_Merchandise trade (% of GDP)'].replace(to_replace = df.loc[[48,84,195], 'out_Merchandise trade (% of GDP)'], value = 1, inplace = True)

    df['out_Mobile cellular subscriptions (per 100 people)'] = 0
    df['out_Mobile cellular subscriptions (per 100 people)'].replace(to_replace = df.loc[[6,27,48], 'out_Mobile cellular subscriptions (per 100 people)'], value = 1, inplace = True)

    df['out_Population ages 0-14 (% of total population)'] = 0
    df['out_Population ages 0-14 (% of total population)'].replace(to_replace = df.loc[[55,84], 'out_Population ages 0-14 (% of total population)'], value = 1, inplace = True)

    df['out_Population ages 15-64 (% of total population)'] = 0
    df['out_Population ages 15-64 (% of total population)'].replace(to_replace = df.loc[[84,211], 'out_Population ages 15-64 (% of total population)'], value = 1, inplace = True)

    df['out_Population ages 65 and above (% of total population)'] = 0
    df['out_Population ages 65 and above (% of total population)'].replace(to_replace = df.loc[[84,158,211], 'out_Population ages 65 and above (% of total population)'], value = 1, inplace = True)

    df['out_Population density (people per sq. km of land area)'] = 0
    df['out_Population density (people per sq. km of land area)'].replace(to_replace = df.loc[[9,16], 'out_Population density (people per sq. km of land area)'], value = 1, inplace = True)

    df['out_Population growth (annual %)'] = 0
    df['out_Population growth (annual %)'].replace(to_replace = df.loc[[158], 'out_Population growth (annual %)'], value = 1, inplace = True)

    df['out_Population, female'] = 0
    df['out_Population, female'].replace(to_replace = df.loc[[48,55,84], 'out_Population, female'], value = 1, inplace = True)

    df['out_Population, male'] = 0
    df['out_Population, male'].replace(to_replace = df.loc[[48,55,84], 'out_Population, male'], value = 1, inplace = True)

    df['out_Population, total'] = 0
    df['out_Population, total'].replace(to_replace = df.loc[[48,55,84], 'out_Population, total'], value = 1, inplace = True)

    df['out_Prevalence of undernourishment (% of population)'] = 0
    df['out_Prevalence of undernourishment (% of population)'].replace(to_replace = df.loc[[84], 'out_Prevalence of undernourishment (% of population)'], value = 1, inplace = True)

    df['out_Services, value added (% of GDP)'] = 0
    df['out_Services, value added (% of GDP)'].replace(to_replace = df.loc[[36,158], 'out_Services, value added (% of GDP)'], value = 1, inplace = True)

    df['out_Surface area (sq. km)'] = 0
    df['out_Surface area (sq. km)'].replace(to_replace = df.loc[[48,55,84], 'out_Surface area (sq. km)'], value = 1, inplace = True)

    df['out_Urban population growth (annual %)'] = 0
    df['out_Urban population growth (annual %)'].replace(to_replace = df.loc[[84,158], 'out_Urban population growth (annual %)'], value = 1, inplace = True)

    return df

def corr_matrix(df):
    """ Defining a function for obtaining a correlation matrix and exporting to excel"""
    corr = df.corr()
    corr.to_excel('Correlation Matrix.xlsx')

def get_region(data, regid='Ratchet'):
    """Filter a region from data"""
    return data[data['Cool Name'] == regid]

def get_columns(data, cols): 
    """Return desired columns from a dataset"""
    data = data.loc[:, cols]
    return data

def get_col_names(data, cols):
    """Get name of a column in a dataframe"""
    names=[]
    for col in cols:
        names += [data.columns[col]]
    return names    

def delete_na(data, cols):
    """Drop null values from a Dataframe column"""
    for col in cols:
        data = data.dropna(axis='index', subset=[col])
    return data

def analize_missing (data, col):
    """Calculate missing values in a column"""
    world        = data
    region       = get_region(data)
    total_world  = len(world.iloc[:,0])
    total_region = len(region.iloc[:,0])
    miss_world   = world.loc [:,col].isnull().sum()    
    miss_region  = region.loc[:,col].isnull().sum()    
    print(col) 
    print(f"""World missing values: {int(miss_world.values)} / {total_world} = {round(int(miss_world.values) / total_world * 100,2)} """)
    print(f"""Region missing values: {int(miss_region.values)} / {total_region} = {round(int(miss_region.values) / total_region * 100,2)}""")
    
    
def show_stats (data, col): 
    """Show statistics for a column for each region"""
    reg_col   = 'Hult Region'
    regions   = data[reg_col].unique()
    describe  = data.loc[:,col].describe().round(2)
    describe.columns = ['Total World']
    for regid in regions: 
        region = data[data[reg_col] == regid]  
        describe[regid]= region.loc[:,col].describe().round(2)
    print(f"\n{describe.transpose()}")

def show_charts_1(data, col):
    """Show histogram and boxplot for a column (World-Region)"""
    world  = delete_na(data, col)
    region = get_region(world)
    
    #Setting for plot
    sns.set()
    sns.set(rc={"figure.figsize": (8, 6)})
    sns.set_style('darkgrid')
    sns.color_palette('BrBG',8)
    
    #World Charts
    plt.subplot(2,2,1)
    sns.distplot(world.loc[:,col], axlabel = 'World')
    plt.subplot(2,2,3)
    sns.boxplot(world.loc[:,col], palette='Set2')  

    #Region Charts
    plt.subplot(2,2,2)
    sns.distplot(region.loc[:,col], axlabel = 'Region')
    plt.subplot(2,2,4)
    sns.boxplot(region.loc[:,col], palette='Set2')

    #Adjust subplots
    plt.tight_layout()
    file = str(col) + "-Charts.jpg"
    plt.savefig(file)
    plt.show()

def show_charts_2(data, col):
    """Show histogram and boxplot after impute data"""
    region = delete_na(data, col)
    region = get_region(region)
    
    #Setting values for plot
    sns.set()
    sns.set(rc={"figure.figsize": (8, 6)})
    sns.set_style('darkgrid')
    sns.color_palette('BrBG',8)
    
    #Region Charts
    plt.subplot(1,2,1)
    sns.distplot(region.loc[:,col], axlabel = 'Region')
    plt.subplot(1,2,2)
    sns.boxplot(region.loc[:,col], palette='Set2')  

    #Adjust subplots
    plt.tight_layout()
    file = str(col) + "-Charts.jpg"
    plt.savefig(file)
    plt.show()

def show_bars(data, col):
    """Show a bar char for each country in a region"""
    world  = delete_na(data, col)
    region = get_region(world)
    sns.set(rc={"figure.figsize": (12, 8)})
    sns.set_style('darkgrid')
    #sns.color_palette('BrBG',8)
    plt.subplots()
    x = col[0]
    y = 'Country Name'
    sns.barplot(x=x, 
                y=y, 
                data=region, 
               )
    mean = region[col].mean()
    plt.plot([mean, mean], [0, len(region)])
    plt.annotate(s='mean', xy=(mean, 0), xytext=(mean+1, 1.0), arrowprops={'color':'red'})
    file = str(col) + "-Country.jpg"
    plt.savefig(file)
    plt.show()

   
def show_regions(data, col):
    """Show histograms for each region"""
    world  = delete_na(data, col)
    region_col = 'Hult Region'
    
    #Setting values for plot
    sns.set(rc={"figure.figsize": (12, 4)})
    sns.set_style('whitegrid')
    sns.color_palette('BrBG',8)
    
    #World Charts
    regions = list(data[region_col].unique())
    del(regions[-1])
    plot = 1
    for val in regions:
        plt.subplot(1, 4, plot)
        region = world[world[region_col] == val]
        sns.distplot(region.loc[:,col], axlabel = val, color='teal')
        plot += 1
        if plot > 4 :
            plt.tight_layout()
            file = str(col) + "-Regions"+str(plot)+".jpg"
            plt.savefig(file)
            plt.show()
            plot = 1
    plt.show()
 

def show_correlation(data, x, y, order):
    """Show a correlation plot between two variables"""
    world  = delete_na(data, x)
    world  = delete_na(data, y)
    region = get_region(world)

    #Global setting for plot
    sns.set(rc={"figure.figsize": (12, 4)})
    sns.set_style('whitegrid')
    sns.color_palette('cubehelix',8)
    
    #World Chart
    g=sns.pairplot(data= world,
                 x_vars= x,
                 y_vars= y,
                 kind='reg', 
                 diag_kind='hist',
                 height=8)
    g.fig.suptitle("World")  
    
    #Region Chart
    g=sns.pairplot(data= region,
                 x_vars= x,
                 y_vars= y,
                 kind='reg', 
                 diag_kind='hist',
                 height=8)
    g.fig.suptitle("Region")
    file = str(x) + str(y) + "Correlation.jpg"
    plt.savefig(file)
    plt.tight_layout()
    plt.show

# A dictionary of types
types = {'Country Code'                                                                                                 :str,
         'Country Name'                                                                                                 :str,
         'Hult Region'                                                                                                  :str,
         'Cool Name'                                                                                                    :str,
         'Access to electricity (% of population)'                                                                      :float,
         'Access to electricity, rural (% of rural population)'                                                         :float,
         'Access to electricity, urban (% of urban population)'                                                         :float,
         'Adolescent fertility rate (births per 1,000 women ages 15-19)'                                                :float,
         'Age dependency ratio (% of working-age population)'                                                           :float,
         'Age dependency ratio, old (% of working-age population)'                                                      :float,
         'Age dependency ratio, young (% of working-age population)'                                                    :float,
         'Agriculture, forestry, and fishing, value added (% of GDP)'                                                   :float,
         'Armed forces personnel (% of total labor force)'                                                              :float,
         'Birth rate, crude (per 1,000 people)'                                                                         :float,
         'Births attended by skilled health staff (% of total)'                                                         :float,
         'Death rate, crude (per 1,000 people)'                                                                         :float,
         'Educational attainment, Doctoral or equivalent, population 25+, total (%) (cumulative)'                       :float,
         'Educational attainment, at least Bachelor\'s or equivalent, population 25+, total (%) (cumulative)'           :float,
         'Educational attainment, at least Master\'s or equivalent, population 25+, total (%) (cumulative)'             :float,
         'Educational attainment, at least completed lower secondary, population 25+, total (%) (cumulative)'           :float,
         'Educational attainment, at least completed post-secondary, population 25+, total (%) (cumulative)'            :float,
         'Educational attainment, at least completed primary, population 25+ years, total (%) (cumulative)'             :float,
         'Educational attainment, at least completed short-cycle tertiary, population 25+, total (%) (cumulative)'      :float,
         'Educational attainment, at least completed upper secondary, population 25+, total (%) (cumulative)'           :float,
         'Employment in agriculture (% of total employment) (modeled ILO estimate)'                                     :float,
         'Employment in industry (% of total employment) (modeled ILO estimate)'                                        :float,
         'Employment in services (% of total employment) (modeled ILO estimate)'                                        :float,
         'Fertility rate, total (births per woman)'                                                                     :float,
         'GDP (current US$)'                                                                                            :float,
         'GDP growth (annual %)'                                                                                        :float,
         'GINI index (World Bank estimate)'                                                                             :float,
         'Government expenditure on education, total (% of government expenditure)'                                     :float,
         'Income share held by fourth 20%'                                                                              :float,
         'Income share held by highest 20%'                                                                             :float,
         'Income share held by lowest 20%'                                                                              :float,
         'Income share held by second 20%'                                                                              :float,
         'Income share held by third 20%'                                                                               :float,
         'Industry (including construction), value added (% of GDP)'                                                    :float,
         'Life expectancy at birth, total (years)'                                                                      :float,
         'Literacy rate, adult total (% of people ages 15 and above)'                                                   :float,
         'Literacy rate, youth total (% of people ages 15-24)'                                                          :float,
         'Merchandise trade (% of GDP)'                                                                                 :float,
         'Military expenditure (% of GDP)'                                                                              :float,
         'Mobile cellular subscriptions (per 100 people)'                                                               :float,
         'Number of people pushed below the $3.10 ($ 2011 PPP) poverty line by out-of-pocket health care expenditure'   :float,
         'Population ages 0-14 (% of total population)'                                                                 :float,
         'Population ages 15-64 (% of total population)'                                                                :float,
         'Population ages 65 and above (% of total population)'                                                         :float,
         'Population density (people per sq. km of land area)'                                                          :float,
         'Population growth (annual %)'                                                                                 :float,
         'Population in the largest city (% of urban population)'                                                       :float,
         'Population living in slums (% of urban population)'                                                           :float,
         'Population, female'                                                                                           :float,
         'Population, male'                                                                                             :float,
         'Population, total'                                                                                            :float,
         'Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)'                                          :float,
         'Poverty headcount ratio at $3.20 a day (2011 PPP) (% of population)'                                          :float,
         'Poverty headcount ratio at national poverty lines (% of population)'                                          :float,
         'Prevalence of HIV, total (% of population ages 15-49)'                                                        :float,
         'Prevalence of undernourishment (% of population)'                                                             :float,
         'Prevalence of underweight, weight for age (% of children under 5)'                                            :float,
         'Rural population (% of total population)'                                                                     :float,
         'Services, value added (% of GDP)'                                                                             :float,
         'Surface area (sq. km)'                                                                                        :float,
         'Tax revenue (% of GDP)'                                                                                       :float,
         'Urban population (% of total population)'                                                                     :float,
         'Urban population growth (annual %)'                                                                           :float,
}

file = 'WDIW Dataset.xlsx'
dataset = pd.read_excel(file,
                     dtype=types)   #read dataset from excel

cool_name = dataset["Cool Name"] == "Ratchet"   #set region name
team4 = dataset.loc[cool_name]                  #filter team group dataset

addsome = GDPcap(team4)                         #Add a calculated column 

flagged = flag_missing(addsome)                 #Flag missing data

added = data_input(flagged)                     #Impute data from different sources

pruned = prune(added)                           #Removing columns with a lot of missing values

imputed = impute_median(pruned)                 #Imputing data with median

cleaned_flagged = flag_outliers(imputed)        #Flagging outliers

cleaned_flagged.to_excel('Final Dataset.xlsx')  #Put on excel a final dataframe

#Exploratory data analysis for a column
col = [26]                             #column number
column = get_col_names(dataset, col)   #column name
analize_missing(dataset, column)       #Report for missing data
show_stats(dataset, column)            #Show statistcs for a column
show_charts_1(dataset, column)         #Show histogram and boxplot
show_bars(dataset, column)             #Show a bar chart per country
show_regions(dataset, column)

#Final analysis for a column
analize_missing(cleaned_flagged, column)      #Report for missing data
show_stats(cleaned_flagged, column)           #Show statistcs for a column
show_charts_2(cleaned_flagged, column)        #Show histogram and boxplot

#Analize correlation between two variables
col1 = get_col_names(dataset, [45])
col2 = get_col_names(dataset, [10])
show_correlation (dataset, y=col2, x=col1, order=2)


# In[ ]:




