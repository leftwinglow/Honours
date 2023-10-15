# Import Camelot dependencies
import ctypes
import tkinter

# Import Camelot PDF parser
import camelot as cm

import pandas as pd

def tbl_convert(file_name):
    tables = cm.read_pdf(f'Raw_Data/{file_name}.pdf', pages = '1-2') #Need 24 pages for this PDF

    # Pandas concatenate Camelot TableList object 'tables' as a Pandas df
    tables_df = pd.concat([
        tables[0].df,
        tables[1].df,
        tables[2].df,
        tables[3].df,
        tables[4].df,
        tables[5].df,
        tables[6].df,
        tables[7].df,
        tables[8].df,
        tables[9].df,
        tables[10].df,
        tables[11].df,
        tables[12].df,
        tables[13].df,
        tables[14].df,
        tables[15].df,
        tables[16].df,
        tables[17].df,
        tables[18].df,
        tables[19].df,
        tables[20].df,
        tables[21].df,
        tables[22].df,
        tables[23].df])

    # Must be opened in Excel via Data > Import CSV > with 65001 UTF encoding for Unicode characters to function
    tables_df.to_csv(f'Transformed_Data/{file_name}.csv', encoding = 'utf_8')


tbl_convert('Onakpoya_Drugs')