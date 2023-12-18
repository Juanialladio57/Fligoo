#%%
import matplotlib.pyplot as plt

#%%

def type_and_missings(df):
    '''
    This function receives a database as input and analyzes the data type and 
    the missing values ​​of each of the variables. Then save the results in a dictionary

    Parameters
    ----------
    df : dataframe
        The pandas DataFrame containing the dataset.
    Returns
    -------
    dictionary : dict
        Is a dictionary that has two keys: data type and missing values. The 
        first saves the report that is obtained when the dtypes method is used.
        The second stores the total number of null values ​​for each variable, 
        resulting from applying isnull().sum()

    '''
    
    dictionary = {
    "data type": df.dtypes,
    "missings values": df.isnull().sum(),
    }
    return dictionary

def descriptive_statistics(df, l, labels):
    '''
    Generate a descriptive statistics table for specified columns in a DataFrame and
    display it as a matplotlib table.

    Parameters
    ----------
    df : DataFrame
        The pandas DataFrame containing the data for which descriptive statistics 
        are to be calculated.
    l : list of str
        The list of column names from the DataFrame for which the descriptive 
        statistics are to be computed.
    labels : list of str
        The list of column labels to be used when displaying the table. These labels 
        should correspond to the columns specified in `l`.

    Returns
    -------
    None
        The function creates a visual table of descriptive statistics and saves it 
        as an image file 'statistics_table.png'. The table is also displayed in the 
        matplotlib viewer.

    Notes
    -----
    The table generated includes statistical measures such as mean, median, count,
    min, max, standard deviation, and quartiles for the specified columns. The 
    function rounds the statistics to 2 decimal places for improved readability.
    '''
    
    statistics_table = df[l].describe() 

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis('off')
    the_table = ax.table(cellText=statistics_table.values.round(2),
                         colLabels=labels,  # Using personalized column labels
                         rowLabels=statistics_table.index,
                         cellLoc='center', 
                         loc='center',
                         bbox=[0, 0, 1, 1])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)  # Ajust font size 
    the_table.auto_set_column_width(col=list(range(len(labels))))  # Adjust column width
    
    # Guardar la figura
    plt.savefig('statistics_table.png', dpi=300)
    plt.show()


