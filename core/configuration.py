from core.dataset import InputTypes, DataTypes, FeatureSpec

class GHLConfig():
    def __init__(self):

        self.features = [
            FeatureSpec('RT_level', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('RT_temperature.T', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('HT_temperature.T', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('inj_valve_act', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('heater_act', InputTypes.TARGET, DataTypes.CONTINUOUS),
        ]
        self.dataset_stride = 1
        self.example_length = 35
        self.encoder_length = 35 #20

        # # Feature sizes
        self.static_categorical_inp_lens = []
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []

        self.n_head = 4
        self.hidden_size = 512
        self.dropout = 0.2
        self.attn_dropout = 0.0

        self.output_size = 5

        self.temporal_known_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                    self.temporal_observed_continuous_inp_size,
                                    self.temporal_target_size,
                                    len(self.temporal_observed_categorical_inp_lens)])

class WADIConfig():
    def __init__(self):

        self.features = [

            FeatureSpec('year', InputTypes.KNOWN, DataTypes.CONTINUOUS),
            FeatureSpec('month', InputTypes.KNOWN, DataTypes.CONTINUOUS),
            FeatureSpec('day', InputTypes.KNOWN, DataTypes.CONTINUOUS),

            
            # FeatureSpec('1_AIT_002_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('1_FIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('1_MV_001_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('1_MV_002_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('1_MV_003_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('1_P_001_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('1_P_003_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('2_FIC_401_SP', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('2_FIC_501_SP', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('2_LT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('2_MCV_007_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('2_MCV_301_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('2_PIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('2_PIT_002_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('TOTAL_CONS_REQUIRED_FLOW', InputTypes.TARGET, DataTypes.CONTINUOUS),

            FeatureSpec('1_AIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_AIT_003_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_AIT_005_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_FIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_LS_001_AL', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_LS_002_AL', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_LT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_MV_001_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_MV_002_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_MV_003_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_MV_004_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_P_001_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_P_002_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_P_003_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_P_004_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_P_005_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_P_006_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_DPIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_101_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_101_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_101_SP', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_201_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_201_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_201_SP', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_301_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_301_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_301_SP', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_401_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_401_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_401_SP', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_501_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_501_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_501_SP', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_601_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_601_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIC_601_SP', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIT_002_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FIT_003_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FQ_101_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FQ_201_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FQ_301_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FQ_401_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FQ_501_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_FQ_601_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_101_AH', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_101_AL', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_201_AH', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_201_AL', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_301_AH', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_301_AL', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_401_AH', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_401_AL', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_501_AH', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_501_AL', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_601_AH', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LS_601_AL', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_LT_002_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MCV_007_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MCV_101_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MCV_201_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MCV_301_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MCV_401_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MCV_501_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MCV_601_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_001_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_002_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_003_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_004_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_005_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_006_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_009_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_101_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_201_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_301_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_401_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_501_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_MV_601_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_P_003_SPEED', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_P_003_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_P_004_SPEED', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_P_004_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_PIC_003_CO', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_PIC_003_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_PIC_003_SP', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_PIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_PIT_002_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_PIT_003_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_SV_101_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_SV_201_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_SV_301_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_SV_401_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_SV_501_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2_SV_601_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2A_AIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2A_AIT_002_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2A_AIT_003_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2A_AIT_004_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2B_AIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            # FeatureSpec('2B_AIT_002_PV', InputTypes.TARGET, DataTypes.CONTINUOUS), #STRANGE DATA
            FeatureSpec('2B_AIT_003_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_AIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_AIT_002_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_AIT_003_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_AIT_005_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_FIT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_LS_001_AL', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_LT_001_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_MV_001_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_MV_002_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_MV_003_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_P_001_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_P_002_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_P_003_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_P_004_STATUS', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('LEAK_DIFF_PRESSURE', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('PLANT_START_STOP_LOG', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('TOTAL_CONS_REQUIRED_FLOW', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_AIT_002_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('1_AIT_004_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('2B_AIT_004_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('3_AIT_004_PV', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('year', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('month', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('day', InputTypes.TARGET, DataTypes.CONTINUOUS),
        ]
        self.dataset_stride = 1
        self.example_length = 14*60#35
        self.encoder_length = 12*60#20
        self.quantiles = [0.1, 0.5, 0.9]

        # # Feature sizes
        self.static_categorical_inp_lens = []
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []

        self.n_head = 4
        self.hidden_size = 256
        self.dropout = 0.2
        self.attn_dropout = 0.0

        self.output_size = 15

        self.temporal_known_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                    self.temporal_observed_continuous_inp_size,
                                    self.temporal_target_size,
                                    len(self.temporal_observed_categorical_inp_lens)])

class SMDConfig():
    def __init__(self):

        self.features = [
            FeatureSpec('time', InputTypes.KNOWN, DataTypes.CONTINUOUS),

            
            FeatureSpec('var_0', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_1', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_2', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_3', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_4', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_5', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_6', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_7', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_8', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_9', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_10', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_11', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_12', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_13', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_14', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_15', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_16', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_17', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_18', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_19', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_20', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_21', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_22', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_23', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_24', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_25', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_26', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_27', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_28', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_29', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_30', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_31', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_32', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_33', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_34', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_35', InputTypes.TARGET, DataTypes.CONTINUOUS),
            FeatureSpec('var_36', InputTypes.TARGET, DataTypes.CONTINUOUS),

        ]
        self.dataset_stride = 1
        self.example_length = 60
        self.encoder_length = 50
        self.quantiles = [0.1, 0.5, 0.9]
        self.nvar = 37

        # # Feature sizes
        self.static_categorical_inp_lens = []
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []

        self.n_head = 4
        self.hidden_size = 256
        self.dropout = 0.2
        self.attn_dropout = 0.0

        self.output_size = 37

        self.temporal_known_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                    self.temporal_observed_continuous_inp_size,
                                    self.temporal_target_size,
                                    len(self.temporal_observed_categorical_inp_lens)])

CONFIGS = {
    'GHL' : GHLConfig,
    'WADI' : WADIConfig,
    'SMD' : SMDConfig
}