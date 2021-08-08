from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
from .context import pandas_ta

from unittest import TestCase, skip
import pandas.testing as pdt
from pandas import DataFrame, Series
import numpy as np

import talib as tal


class TestVolume(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        if "volume" in cls.data.columns:
            cls.volume_ = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        if hasattr(cls, "volume"):
            del cls.volume_
        del cls.data

    def setUp(self): pass
    def tearDown(self): pass


    def test_ad(self):
        result = pandas_ta.ad(self.high, self.low, self.close, self.volume_, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "AD")

        try:
            expected = tal.AD(self.high, self.low, self.close, self.volume_)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.ad(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "AD")

    def test_ad_open(self):
        result = pandas_ta.ad(self.high, self.low, self.close, self.volume_, self.open)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ADo")

    def test_adosc(self):
        result = pandas_ta.adosc(self.high, self.low, self.close, self.volume_, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ADOSC_3_10")

        try:
            expected = tal.ADOSC(self.high, self.low, self.close, self.volume_)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.adosc(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ADOSC_3_10")

    def test_aobv(self):
        result = pandas_ta.aobv(self.close, self.volume_)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AOBVe_4_12_2_2_2")

    def test_cmf(self):
        result = pandas_ta.cmf(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CMF_20")

    def test_efi(self):
        result = pandas_ta.efi(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EFI_13")

    def test_eom(self):
        result = pandas_ta.eom(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EOM_14_100000000")

    def test_kvo(self):
        result = pandas_ta.kvo(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KVO_34_55_13")

    def test_mfi(self):
        result = pandas_ta.mfi(self.high, self.low, self.close, self.volume_, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MFI_14")

        try:
            expected = tal.MFI(self.high, self.low, self.close, self.volume_)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.mfi(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MFI_14")

    def test_nvi(self):
        result = pandas_ta.nvi(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "NVI_1")

    def test_obv(self):
        result = pandas_ta.obv(self.close, self.volume_, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "OBV")

        try:
            expected = tal.OBV(self.close, self.volume_)
            pdt.assert_series_equal(result, expected, check_names=False)
        except AssertionError:
            try:
                corr = pandas_ta.utils.df_error_analysis(result, expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = pandas_ta.obv(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "OBV")

    def test_pvi(self):
        result = pandas_ta.pvi(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVI_1")

    def test_pvol(self):
        result = pandas_ta.pvol(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVOL")

    def test_pvr(self):
        result = pandas_ta.pvr(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVR")
        # sample indicator values from SPY
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 3)
        self.assertEqual(result[4], 2)
        self.assertEqual(result[6], 4)

    def test_pvt(self):
        result = pandas_ta.pvt(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVT")

    def test_vp(self):
        result = pandas_ta.vp(self.close, self.volume_)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "VP_10")

    def test_vp_extended_dual_volume(self):
        """Volume Profile for dual volume"""
        df = pandas_ta.vp_extended(self.data[:30])

        """
        Expected Result Table:
        	Lower Price	Upper Price	Total Vol	Mean Price			
        0	134.583	    135.519	    13739200	135.04685			PL
        1	135.519	    136.444	    4006500	    135.56250			
        2	136.444	    137.369	    12441200	136.61715			
        3	137.369	    138.294	    18486300	137.86457			
        4	138.294	    139.219	    4794100	    138.50000			
        5	139.219	    140.144	    21672400	139.70310	    2	VAL
        6	140.144	    141.069	    18940200	140.68747	    2	
        7	141.069	    141.994	    58089900	141.51734	    1	POC
        8	141.994	    142.919	    20828200	142.49998	    3	
        9	142.919	    143.844	    10045400	143.84370	    3	VAH/PH
        """
        """
        Columns:
        ['price_low', 'price_mean', 'price_high', 'volume_total', 'volume_neg',
       'volume_pos', 'position', 'profile_low', 'profile_high',
       'point_of_control', 'value_area_low', 'value_area_high']
        """
        
        self.assertEquals(df['price_low'][0],134.583)
        self.assertEquals(df['price_low'][1],135.519)
        self.assertEquals(df['price_low'][2],136.444)
        self.assertEquals(df['price_low'][3],137.369)
        self.assertEquals(df['price_low'][4],138.294)
        self.assertEquals(df['price_low'][5],139.219)
        self.assertEquals(df['price_low'][6],140.144)
        self.assertEquals(df['price_low'][7],141.069)
        self.assertEquals(df['price_low'][8],141.994)
        self.assertEquals(df['price_low'][9],142.919)

        self.assertEquals(df['price_high'][0],135.519)
        self.assertEquals(df['price_high'][1],136.444)
        self.assertEquals(df['price_high'][2],137.369)
        self.assertEquals(df['price_high'][3],138.294)
        self.assertEquals(df['price_high'][4],139.219)
        self.assertEquals(df['price_high'][5],140.144)
        self.assertEquals(df['price_high'][6],141.069)
        self.assertEquals(df['price_high'][7],141.994)
        self.assertEquals(df['price_high'][8],142.919)
        self.assertEquals(df['price_high'][9],143.844)

        self.assertEquals(float("{:.5f}".format(df['price_mean'][0])),135.04685)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][1])),135.56250)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][2])),136.61715)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][3])),137.86457)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][4])),138.50000)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][5])),139.70310)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][6])),140.68747)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][7])),141.51734)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][8])),142.49998)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][9])),143.84370)

        self.assertEquals(df['volume_total'][0],13739200)
        self.assertEquals(df['volume_total'][1],4006500)
        self.assertEquals(df['volume_total'][2],12441200)
        self.assertEquals(df['volume_total'][3],18486300)
        self.assertEquals(df['volume_total'][4],4794100)
        self.assertEquals(df['volume_total'][5],21672400)
        self.assertEquals(df['volume_total'][6],18940200)
        self.assertEquals(df['volume_total'][7],58089900)
        self.assertEquals(df['volume_total'][8],20828200)
        self.assertEquals(df['volume_total'][9],10045400)

        self.assertTrue(np.isnan(df['position'][0]))
        self.assertTrue(np.isnan(df['position'][1]))
        self.assertTrue(np.isnan(df['position'][2]))
        self.assertTrue(np.isnan(df['position'][3]))
        self.assertTrue(np.isnan(df['position'][4]))
        self.assertEquals(df['position'][5],2)
        self.assertEquals(df['position'][6],2)
        self.assertEquals(df['position'][7],1)
        self.assertEquals(df['position'][8],3)
        self.assertEquals(df['position'][9],3)

        self.assertEquals(df['profile_low'][0],1)
        self.assertEquals(df['profile_low'][1],0)
        self.assertEquals(df['profile_low'][2],0)
        self.assertEquals(df['profile_low'][3],0)
        self.assertEquals(df['profile_low'][4],0)
        self.assertEquals(df['profile_low'][5],0)
        self.assertEquals(df['profile_low'][6],0)
        self.assertEquals(df['profile_low'][7],0)
        self.assertEquals(df['profile_low'][8],0)
        self.assertEquals(df['profile_low'][9],0)

        self.assertEquals(df['profile_high'][0],0)
        self.assertEquals(df['profile_high'][1],0)
        self.assertEquals(df['profile_high'][2],0)
        self.assertEquals(df['profile_high'][3],0)
        self.assertEquals(df['profile_high'][4],0)
        self.assertEquals(df['profile_high'][5],0)
        self.assertEquals(df['profile_high'][6],0)
        self.assertEquals(df['profile_high'][7],0)
        self.assertEquals(df['profile_high'][8],0)
        self.assertEquals(df['profile_high'][9],1)

        self.assertEquals(df['point_of_control'][0],0)
        self.assertEquals(df['point_of_control'][1],0)
        self.assertEquals(df['point_of_control'][2],0)
        self.assertEquals(df['point_of_control'][3],0)
        self.assertEquals(df['point_of_control'][4],0)
        self.assertEquals(df['point_of_control'][5],0)
        self.assertEquals(df['point_of_control'][6],0)
        self.assertEquals(df['point_of_control'][7],1)
        self.assertEquals(df['point_of_control'][8],0)
        self.assertEquals(df['point_of_control'][9],0)

        self.assertEquals(df['value_area_low'][0],0)
        self.assertEquals(df['value_area_low'][1],0)
        self.assertEquals(df['value_area_low'][2],0)
        self.assertEquals(df['value_area_low'][3],0)
        self.assertEquals(df['value_area_low'][4],0)
        self.assertEquals(df['value_area_low'][5],1)
        self.assertEquals(df['value_area_low'][6],0)
        self.assertEquals(df['value_area_low'][7],0)
        self.assertEquals(df['value_area_low'][8],0)
        self.assertEquals(df['value_area_low'][9],0)

        self.assertEquals(df['value_area_high'][0],0)
        self.assertEquals(df['value_area_high'][1],0)
        self.assertEquals(df['value_area_high'][2],0)
        self.assertEquals(df['value_area_high'][3],0)
        self.assertEquals(df['value_area_high'][4],0)
        self.assertEquals(df['value_area_high'][5],0)
        self.assertEquals(df['value_area_high'][6],0)
        self.assertEquals(df['value_area_high'][7],0)
        self.assertEquals(df['value_area_high'][8],0)
        self.assertEquals(df['value_area_high'][9],1)

    def test_vp_extended_single_volume(self):
        """Volume Profile for single volume"""
        df = pandas_ta.vp_extended(self.data[:30], nr_volumes=1)

        """
        Expected Result Table:
        	Lower Price	Upper Price	Total Vol	Mean Price			
        0	134.583	    135.519	    13739200	135.04685			PL
        1	135.519	    136.444	    4006500	    135.56250			
        2	136.444	    137.369	    12441200	136.61715			
        3	137.369	    138.294	    18486300	137.86457			
        4	138.294	    139.219	    4794100	    138.50000			
        5	139.219	    140.144	    21672400	139.70310	    4	VAL
        6	140.144	    141.069	    18940200	140.68747	    3	
        7	141.069	    141.994	    58089900	141.51734	    1	POC
        8	141.994	    142.919	    20828200	142.49998	    2	
        9	142.919	    143.844	    10045400	143.84370	    5	VAH/PH
        """
        """
        Columns:
        ['price_low', 'price_mean', 'price_high', 'volume_total', 'volume_neg',
       'volume_pos', 'position', 'profile_low', 'profile_high',
       'point_of_control', 'value_area_low', 'value_area_high']
        """
        
        print(df.head(10))

        self.assertEquals(df['price_low'][0],134.583)
        self.assertEquals(df['price_low'][1],135.519)
        self.assertEquals(df['price_low'][2],136.444)
        self.assertEquals(df['price_low'][3],137.369)
        self.assertEquals(df['price_low'][4],138.294)
        self.assertEquals(df['price_low'][5],139.219)
        self.assertEquals(df['price_low'][6],140.144)
        self.assertEquals(df['price_low'][7],141.069)
        self.assertEquals(df['price_low'][8],141.994)
        self.assertEquals(df['price_low'][9],142.919)

        self.assertEquals(df['price_high'][0],135.519)
        self.assertEquals(df['price_high'][1],136.444)
        self.assertEquals(df['price_high'][2],137.369)
        self.assertEquals(df['price_high'][3],138.294)
        self.assertEquals(df['price_high'][4],139.219)
        self.assertEquals(df['price_high'][5],140.144)
        self.assertEquals(df['price_high'][6],141.069)
        self.assertEquals(df['price_high'][7],141.994)
        self.assertEquals(df['price_high'][8],142.919)
        self.assertEquals(df['price_high'][9],143.844)

        self.assertEquals(float("{:.5f}".format(df['price_mean'][0])),135.04685)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][1])),135.56250)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][2])),136.61715)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][3])),137.86457)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][4])),138.50000)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][5])),139.70310)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][6])),140.68747)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][7])),141.51734)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][8])),142.49998)
        self.assertEquals(float("{:.5f}".format(df['price_mean'][9])),143.84370)

        self.assertEquals(df['volume_total'][0],13739200)
        self.assertEquals(df['volume_total'][1],4006500)
        self.assertEquals(df['volume_total'][2],12441200)
        self.assertEquals(df['volume_total'][3],18486300)
        self.assertEquals(df['volume_total'][4],4794100)
        self.assertEquals(df['volume_total'][5],21672400)
        self.assertEquals(df['volume_total'][6],18940200)
        self.assertEquals(df['volume_total'][7],58089900)
        self.assertEquals(df['volume_total'][8],20828200)
        self.assertEquals(df['volume_total'][9],10045400)

        self.assertTrue(np.isnan(df['position'][0]))
        self.assertTrue(np.isnan(df['position'][1]))
        self.assertTrue(np.isnan(df['position'][2]))
        self.assertTrue(np.isnan(df['position'][3]))
        self.assertTrue(np.isnan(df['position'][4]))
        self.assertEquals(df['position'][5],4)
        self.assertEquals(df['position'][6],3)
        self.assertEquals(df['position'][7],1)
        self.assertEquals(df['position'][8],2)
        self.assertEquals(df['position'][9],5)

        self.assertEquals(df['profile_low'][0],1)
        self.assertEquals(df['profile_low'][1],0)
        self.assertEquals(df['profile_low'][2],0)
        self.assertEquals(df['profile_low'][3],0)
        self.assertEquals(df['profile_low'][4],0)
        self.assertEquals(df['profile_low'][5],0)
        self.assertEquals(df['profile_low'][6],0)
        self.assertEquals(df['profile_low'][7],0)
        self.assertEquals(df['profile_low'][8],0)
        self.assertEquals(df['profile_low'][9],0)

        self.assertEquals(df['profile_high'][0],0)
        self.assertEquals(df['profile_high'][1],0)
        self.assertEquals(df['profile_high'][2],0)
        self.assertEquals(df['profile_high'][3],0)
        self.assertEquals(df['profile_high'][4],0)
        self.assertEquals(df['profile_high'][5],0)
        self.assertEquals(df['profile_high'][6],0)
        self.assertEquals(df['profile_high'][7],0)
        self.assertEquals(df['profile_high'][8],0)
        self.assertEquals(df['profile_high'][9],1)

        self.assertEquals(df['point_of_control'][0],0)
        self.assertEquals(df['point_of_control'][1],0)
        self.assertEquals(df['point_of_control'][2],0)
        self.assertEquals(df['point_of_control'][3],0)
        self.assertEquals(df['point_of_control'][4],0)
        self.assertEquals(df['point_of_control'][5],0)
        self.assertEquals(df['point_of_control'][6],0)
        self.assertEquals(df['point_of_control'][7],1)
        self.assertEquals(df['point_of_control'][8],0)
        self.assertEquals(df['point_of_control'][9],0)

        self.assertEquals(df['value_area_low'][0],0)
        self.assertEquals(df['value_area_low'][1],0)
        self.assertEquals(df['value_area_low'][2],0)
        self.assertEquals(df['value_area_low'][3],0)
        self.assertEquals(df['value_area_low'][4],0)
        self.assertEquals(df['value_area_low'][5],1)
        self.assertEquals(df['value_area_low'][6],0)
        self.assertEquals(df['value_area_low'][7],0)
        self.assertEquals(df['value_area_low'][8],0)
        self.assertEquals(df['value_area_low'][9],0)

        self.assertEquals(df['value_area_high'][0],0)
        self.assertEquals(df['value_area_high'][1],0)
        self.assertEquals(df['value_area_high'][2],0)
        self.assertEquals(df['value_area_high'][3],0)
        self.assertEquals(df['value_area_high'][4],0)
        self.assertEquals(df['value_area_high'][5],0)
        self.assertEquals(df['value_area_high'][6],0)
        self.assertEquals(df['value_area_high'][7],0)
        self.assertEquals(df['value_area_high'][8],0)
        self.assertEquals(df['value_area_high'][9],1)