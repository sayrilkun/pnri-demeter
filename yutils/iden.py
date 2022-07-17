from yutils.imports import *
from kivymd.uix.list import IRightBodyTouch, OneLineListItem
from yutils.pyre import db
class OneLine(OneLineListItem):
    divider = None

def identify(self,morph):
    # CONVERT LIST TO DF
    self.help.get_screen('result').ids.passed.clear_widgets()
    morph = {'Morphology': {'sample 1': {'Caudicle Bulb': '0.08', 'Extension': '0.13', 'Hips': '0.34', 'Pollinium Length': '0.73', 'Pollinium Widest': '0.38', 'Retinaculum Length': '0.20', 'Shoulder': '0.08', 'Translator Arm Length': '0.16', 'Translator Depth': '0.06', 'Waist': '0.13'}, 'sample 2': {'Caudicle Bulb': '0.06', 'Extension': '0.11', 'Hips': '0.06', 'Pollinium Length': '0.81', 'Pollinium Widest': '0.19', 'Retinaculum Length': '0.61', 'Shoulder': '0.17', 'Translator Arm Length': '0.23', 'Translator Depth': '0.05', 'Waist': '0.12'}, 'sample 3': {'Caudicle Bulb': '0.12', 'Extension': '0.17', 'Hips': '0.02', 'Pollinium Length': '0.62', 'Pollinium Widest': '0.24', 'Retinaculum Length': '0.35', 'Shoulder': '0.26', 'Translator Arm Length': '0.03', 'Translator Depth': '0.02', 'Waist': '0.08'}}}

    sample_name = []
    sample_data=[]
    for key, value in morph['Morphology'].items():
        sample_name.append(key)
    for j in sample_name:
        array2=[]
        for key, value in morph['Morphology'][j].items():
            array2.append(float(value))
        sample_data.append(array2)
    uk=pd.DataFrame(sample_data)


    # GET MEAN
    uk_mean = uk.mean()
    uk_mean_df = pd.DataFrame(uk_mean).T

    # GET DATA OF SPECIES WITH <3 
    less_data = []
    names = []
    hey = db.child("Hoya").order_by_child("sample_num").end_at(2).get()
    for user in hey.each():
        names.append(user.key())

    for i in names:
        array = []
        less_sample = []
        less_morphology = (db.child("Hoya").child(i).child("Morphology").get())
        for sample_name in less_morphology.each():
            less_sample.append(sample_name.key())

        for j in range (len(less_sample)):
            print(less_sample[j])
            sample_values = db.child("Hoya").child(i).child("Morphology").child(less_sample[j]).get()
            for datas in sample_values.each():
                array.append(float(datas.val()))

        less_data.append(array)
    dd = pd.DataFrame(less_data)

    # GET DATA OF SPECIES WITH >=3 SAMPLES
    names2 = []
    hey = db.child("Hoya").order_by_child("sample_num").start_at(3).get()
    for user in hey.each():
        names2.append(user.key())

    d = {}
    for i in names2:
        more_array2 = []
        more_sample2 = []
        more_morphology2 = (db.child("Hoya").child(i).child("Morphology").get())
        for sample_name in more_morphology2.each():
            more_sample2.append(sample_name.key())
        # print(sample2)

        for j in more_sample2:
            more_array3=[]
            sample_values = db.child("Hoya").child(i).child("Morphology").child(j).get()
            for datas in sample_values.each():
                # print(datas.val())
                more_array3.append(float(datas.val()))
            more_array2.append(more_array3)
        d[i] = pd.DataFrame(more_array2)

    # MERGE LESS AND MORE
    po = {}
    # er = {}
    for i in names2:
        po[i] = d[i].mean()
        er = pd.DataFrame(po).T

    merge =  pd.concat([dd,er], axis = 0)
    mer = merge.set_axis([x for x in range(len(merge))], axis=0)


    # GET DATA OF MEAN TO MATCH DB

    matched_input = []
    input1 = []

    for i in uk_mean_df.values:
        for j in i:
            input1.append(j)

    for i in range(len(merge)):
        matched_input.append(input1)
    idf = pd.DataFrame(matched_input)


    # GET DIFFERENCE SCORES
    score = abs((idf-mer)/((idf+mer)/2))*100
    score['mean difference score'] = round(score.mean(axis=1),2)
    con = np.concatenate((names, names2)) #join array
    score['species'] = con
    score_table = score[['species','mean difference score']] #display selected columns
    
    # DETERMINE WHAT SPECIES PASSED
    passed = score['mean difference score'][score['mean difference score'] <= 5].index.tolist()
    passed_species = score_table['species'][passed].values
    print(passed_species)

    for passer in passed_species:

        self.help.get_screen('result').ids.passed.add_widget(
            OneLine(
                text= passer,
                on_press= lambda x, value = passer : self.ttest_dialog(uk,passed_species,d,value)
            )
        )

    # if len(passed_species) > 0:
    # #T TEST




    #     t_test_val = []
    #     array_t_test = []
    #     landmarks = ['Caudicle Bulb', 
    #                 "Extension",
    #                 "Hips",
    #                 'Pollinium Length',
    #                 "Pollinium Widest",
    #                 "Retinaculum Length",
    #                 "Shoulder",
    #                 "Translator Arm Length",
    #                 "Translator Depth",
    #                 "Waist",]
            
    #     for i in range(len(uk.T)):
    #         val = ttest_ind(d[passer][i], uk[i])
    #         array_t_test.append(round(val[1],2))
    #         # print(val[1])
    #     t_test_val.append(array_t_test)
    #     # t_test_val

    #     print(t_test_val)

    #     t_test_df = pd.DataFrame(t_test_val).T
    #     t_test_df.insert(0,"landmarks",landmarks) 
        
    #     conditions = [
    #         (t_test_df[0] <= 0.05),
    #         (t_test_df[0] > 0.05)
    #         ]
    #     values = ['significant difference', 'no significant difference']

    #     t_test_df['interpretaion'] = np.select(conditions, values)
    #     print(score_table, passed_species, t_test_df)


    #     t_test_row = list(t_test_df.itertuples(index=False, name=None))
    #     print(t_test_row)
    #     # layout = AnchorLayout()
    #     data_tables = MDDataTable(
    #         size_hint=(0.9, 0.9),
    #         use_pagination=True,
    #         rows_num=10,
    #         column_data=[
    #             ("Landmarks", dp(35)),
    #             ("pvalue", dp(15)),
    #             ("interpretation", dp(50)),
    #         ],
    #         row_data= t_test_row
    #     )
    #     self.help.get_screen('result').ids.ttest.add_widget(data_tables) 
        # self.content_cls.dialog8.ids.ttest.add_widget(data_tables)

        # self.help.get_screen('result').ids.passed.add_widget(
        #     OneLine(
        #         text= passer,
        #         on_press = lambda x: self.ttest_dialog()
        #     )
        # )


    score_row = list(score_table.itertuples(index=False, name=None))
    print(score_row)
    self.dialog6.dismiss(force=True)
    self.swtchScreen('result')

    # layout = AnchorLayout()
    data_tables = MDDataTable(
        size_hint=(0.9, 0.9),
        use_pagination=True,
        rows_num=10,
        column_data=[
            ("Hoya Species", dp(35)),
            ("Mean Difference Score", dp(50)),
        ],
        row_data= score_row
    )
    self.help.get_screen('result').ids.diff.add_widget(data_tables)

   
        # self.dialog6.dismiss(force=True)













    # else:

    #     score_row = list(score_table.itertuples(index=False, name=None))
    #     print(score_row)
    #     # layout = AnchorLayout()
    #     data_tables = MDDataTable(
    #         size_hint=(0.5, 0.5),
    #         use_pagination=True,
    #         rows_num=10,
    #         column_data=[
    #             ("Hoya Species", dp(35)),
    #             ("Mean Difference Score", dp(50)),
    #         ],
    #         row_data= score_row
    #     )
    #     self.help.get_screen('result').ids.diff.add_widget(data_tables)
        # self.dialog6.dismiss(force=True)

    # self.help.get_screen('uploaddoc').current
    # help.current = 'camera'

    
    


