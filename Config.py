class Options:
    def __init__(self):
        # model parameters
        self.max_g = 10    # Maximum gap: Controls recency and gap between states
        self.sup_pos = 0.2 # Minimum support in shock group (store_pattern - main.py)
        self.sup_neg = 0.2 # Minimum support in non-shock group (store_pattern - main.py)

        # experiment settings
        self.early_prediction = 24      # For'left' alignment, early_prediction = the time window starting from the begnning 
        self.observation_window = 7*24  # *** Check the length of trajectories
        self.alignment = 'right' 
        self.settings = 'trunc'
        self.num_folds = 10

        # directory settings
        self.ts_pos_filepath = 'mimic/mimic_shock.csv'
        self.ts_neg_filepath = 'mimic/mimic_nonshock.csv'

        self.res_path = 'results/'
        self.patterns_path = 'patterns/'

        '''
        Available Classifiers:
            SVM: 'svm'
            Decision Tree: 'dt'
            Naive Bayes: 'nb'
            Logistic Regression: 'lr'
            K-Nearest Neighbors: 'knn'
        '''
        self.classifier = ''
        #self.classifier = 'lr'
        self.lr_settings = ''
        self.kernl = ''
        # number of temporal patterns: {2: before, o-occur, 3: before, overlap, contain}
        self.num_tp_rel = 2

        # Dataset features specification
        # Exclude 'Procalcitonin' because it has only nulls 
        self.all_feat = ['SystolicBP', 'DiastolicBP', 'HeartRate', 'RespiratoryRate', 'Temperature', 'MAP', \
                         'PulseOx', 'FIO2', 'OxygenFlow', 'Procalcitonin', 'WBC', 'Bands', 'BUN', 'Lactate', \
                         'Platelet', 'Creatinine', 'BiliRubin', 'CReactiveProtein', 'SedRate']
        self.vitals = ['SystolicBP', 'DiastolicBP', 'HeartRate', 'RespiratoryRate', 'Temperature', 'MAP', 'PulseOx']
        self.labs = ['Procalcitonin', 'WBC', 'Bands', 'BUN', 'Lactate', \
                         'Platelet', 'Creatinine', 'BiliRubin', 'CReactiveProtein', 'SedRate']
        self.oxygenCtrl = ['FIO2', 'OxygenFlow']
        if True: # Use a small set of features for efficiency in this workshop        
            self.numerical_feat = ['SystolicBP','HeartRate','RespiratoryRate','Temperature', 'WBC']
            
        else: # Use all numerical features
            self.numerical_feat = ['SystolicBP','DiastolicBP','HeartRate','RespiratoryRate','Temperature','PulseOx','BUN',\
                                'WBC' ,'Bands' ,'Lactate' ,'Platelet','Creatinine','MAP','BiliRubin','CReactiveProtein',\
                                'SedRate' ,'FIO2','OxygenFlow']
            


        self.num_categories = ['VL', 'L', 'N', 'H', 'VH']
        self.binary_feat = [] # ['Observed_InfectionFlag', 'InflammationFlag', 'OrganFailure']
        self.categorical_feat = [] #{'CurrentLocationTypeCode':['ED', 'NURSE', 'ICU', 'STEPDN']}
        self.unique_id_variable = 'VisitIdentifier'
        self.timestamp_variable = 'MinutesFromArrival'

