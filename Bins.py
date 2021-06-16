import pandas as pd

def bins(row,col):
    if row[col] < 100:
        val = 1
    elif row[col]>= 100   and row[col]< 150 :
        val = 2
    elif row[col] >= 150 and row[col] < 200 :
        val = 3
    elif row[col] >=200 and row[col] < 300 :
        val = 4
    elif row[col] >= 300 and row[col] < 449:
        val = 5
    else:
        val = 6
    return val

def bins_reop(row,col):
    if row[col] < 10:
        val = 1
    elif row[col]>= 10   and row[col]< 20 :
        val = 2
    elif row[col] >= 20 and row[col] < 30 :
        val = 3
    else:
        val = 4
    return val


df_hosp = pd.read_csv("Draft/hospid_allyears_expec_hospid_STSRCHOSPD.csv")
df_hosp['bin_total_cardiac'] = df_hosp.apply(bins, col='total surgery count', axis=1)
df_hosp['bin_total_CABG'] = df_hosp.apply(bins, col='total', axis=1)
df_hosp['bin_Reop_CABG'] = df_hosp.apply(bins_reop, col='Reop', axis=1)
df_hosp.to_csv("hospid_allyears_expec_hospid_STSRCHOSPD_div.csv")

df_surg = pd.read_csv("Draft/surgid_allyears_expec_surgid_STSRCHOSPD.csv")
df_surg["bin_total_cardiac"] = df_surg.apply(bins, col="total cardiac surgery", axis=1)
df_surg['bin_total_CABG'] = df_surg.apply(bins, col='total', axis=1)
df_surg['bin_Reop_CABG'] = df_surg.apply(bins_reop, col='Reop', axis=1)
df_surg.to_csv("surgid_allyears_expec_surgid_STSRCHOSPD_div.csv")


df_hosp = pd.read_csv("Draft/hospid_allyears_expec_hospid_STSRCOM.csv")
df_hosp['bin_total_cardiac'] = df_hosp.apply(bins, col='total surgery count', axis=1)
df_hosp['bin_total_CABG'] = df_hosp.apply(bins, col='total', axis=1)
df_hosp['bin_Reop_CABG'] = df_hosp.apply(bins_reop, col='Reop', axis=1)
df_hosp.to_csv("hospid_allyears_expec_hospid_STSRCOM_div.csv")


df_surg = pd.read_csv("Draft/surgid_allyears_expec_surgid_STSRCOM.csv")
df_surg["bin_total_cardiac"] = df_surg.apply(bins, col="total cardiac surgery", axis=1)
df_surg['bin_total_CABG'] = df_surg.apply(bins, col='total', axis=1)
df_surg['bin_Reop_CABG'] = df_surg.apply(bins_reop, col='Reop', axis=1)
df_surg.to_csv("surgid_allyears_expec_surgid_STSRCOM_div.csv")


df_hosp = pd.read_csv("hospid_allyears_expec_hospid_STSRCMM.csv")
df_hosp["hospbin_total_cardiac"] = df_hosp.apply(bins, col="total surgery count", axis=1)
df_hosp['bin_total_CABG'] = df_hosp.apply(bins, col='total', axis=1)
df_hosp['bin_Reop_CABG'] = df_hosp.apply(bins_reop, col='Reop', axis=1)
df_hosp.to_csv("hospid_allyears_expec_hospid_STSRCMM_div.csv")



df_surg = pd.read_csv("surgid_allyears_expec_surgid_STSRCMM.csv")
df_surg["bin_total_cardiac"] = df_surg.apply(bins, col="total cardiac surgery", axis=1)
df_surg['bin_total_CABG'] = df_surg.apply(bins, col='total', axis=1)
df_surg['bin_Reop_CABG'] = df_surg.apply(bins_reop, col='Reop', axis=1)
df_surg.to_csv("surgid_allyears_expec_surgid_STSRCMM_div.csv")
