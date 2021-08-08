"""
Date: 2021-08-08
Name: m4rc3l-h3

Module for creating a Volume Profile and Value Area following procedure
defined in Dalton et al. (2013): Mind Over Markets. 2nd Ed, Wiley.

A Volume Profile is a representation of the volume traded at a given price
over a certain period. In comparison to traditional volume charts, the Volume
Profile is not is not time but price bound.
The Value Area is the price range in a Volume Profile in which a certain
percentage of the overall volume is traded.

Additional Resources:
- https://www.tradingview.com/support/solutions/43000502040-volume-profile/
- https://tradeproacademy.com/how-to-trade-stocks-with-volume-profile-strategy/
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Tuple 

def get_price_volume(
    np_volume : np.array, 
    start_index : int,
    np_pos: np.array,
    lower_direction : bool = True,
    nr_volumes : int = 2
    ) -> Tuple:
    """
    Returns sum of volume and indices for bins in volume profile
    table, starting from Point of Control (=start_index) next to 
    bins already considered in calculation of value area (=bins
    have already been asigned a position in np_pos).

    Args:
        np_volume(np.array): Total volume of bins
        start_index(int): Index of Point of Control
        np_pos(np.array): Position in Value Area
        lower_direction(bool): True is searching for lower price bins (default), False otherwise.
        nr_volumes(int): Number of bins to aggregate (default: 2)
    Returns:
        Tuple with: 
        [0] List[int]: indices of bins 
        [1] int: aggregated volume
        [2] bool: True if profile boundary in direction is reached
    Raises:
        ValueError
    """

    #Validate arguments
    if np_volume is None or np_pos is None: raise ValueError('np_volume and np_pos must not be none.')
    if not np_volume.shape[0] == np_pos.shape[0]: raise ValueError('np_volume and np_pos must be of same size.')
    if start_index < 0 or start_index >= np_volume.shape[0]: raise ValueError('Index of Point of Control not in boundaries.')
    if nr_volumes > np_volume.shape[0]-1: raise ValueError('Number of bins to aggregate exceeds number of bins.')

    #Prepare indices in case of upper direction search
    indices = np.arange(np_volume.shape[0], dtype=int)
    if not lower_direction:
        #Reverse if searching in upper direction of profile
        np_volume = np.flip(np_volume)
        np_pos = np.flip(np_pos)
        indices = np.flip(indices)
        start_index = indices[start_index]

    start_index -= 1
    while (start_index >= 0):
        if np.isnan(np_pos[start_index]):
            #Found free positions
            volume = 0
            ret_indices = []
            boundary = False
            for i in range(start_index, start_index-nr_volumes, -1):
                if i >= 0:
                    volume += np_volume[i]
                    ret_indices.append(indices[i])
                else:
                    #Could not add total number of bins
                    boundary = True
                    break
            return ret_indices, volume, boundary
        elif start_index == 0:
            #No free positions in current direction
            return np.nan, np.nan, True
        else:
            start_index -= 1

def set_up(
    df : pd.DataFrame, 
    nr_bins : int = 10, 
    price_col : str = 'close'
    ) -> Dict:
    """
    Creating prerequisites for determination of volume profile and
    value area by dividing price and volume data into the number of bins
    specified by nr_bins.

    Args:
        df(pd.DataFrame): DataFrame containing volume and price column (see price_col)
        nr_bins(int): Number of bins data should be grouped in (default: 10)
        price_col(str): name of price column in DataFrame (default: close)
    Returns:
        Features generated as dict with keys and respective np.arrays with length of nr_bins:
        -price_mean: average price in bin
        -volume_pos: positive volume in bin (if price increased to previous period)
        -volume_neg: negative volume in bin (if price decresed to previsous period)
        -price_low: lower price boundary of bin (excluding)
        -price_high: upper price boundary of bin (including)
        -volume_total: total volume in bin
    Raises:
        ValueError if arguments are invalid.
    """
    #Validate arguments
    if price_col is None or price_col == '': raise ValueError('Price column in DataFrame not set.')
    if nr_bins <= 0 or nr_bins >= df.shape[0]: raise ValueError('nr_bins must not be 0 or greater than size of DataFrame.')
    if not price_col in df.columns or not 'volume' in df.columns: raise ValueError('DataFrame does not contain {0} or volume column.'.format(price_col))

    signed_price = ta.signed_series(df[price_col], 1)
    df['volume_pos'] = df.volume * signed_price[signed_price > 0]
    df['volume_neg'] = df.volume * - signed_price[signed_price < 0]
    df['price_mean'] = df[price_col].copy()

    #Calculation of bins by cut
    #bin_size = (x[-1] - x[0]) / nr_bins
    #lower = format(x[0] - ((x[-1] - x[0]) * 0.001 + 1 / pow(10,precision)), '.{0}f'.format(precision))
    #print ('First: ({0}, {1}]'.format(lower_bound, x[0] + bin_size))
    #print ('Last: ({0}, {1}]'.format(x[0]+(nr_bins-1)*bin_size, x[0]+(nr_bins)*bin_size))
    vpdf = df.groupby(pd.cut(df[price_col], nr_bins, include_lowest=True, precision=3)).agg({
            'price_mean' : np.mean,
            'volume_pos' : sum,
            'volume_neg': sum,
        })
    vpdf['lower_b'] = [x.left for x in vpdf.index]
    vpdf['upper_b'] = [x.right for x in vpdf.index]
    vpdf['volume_total'] = vpdf.volume_pos + vpdf.volume_neg 
    vpdf = vpdf.reset_index(drop=True)
    
    #Prepare return
    temp = vpdf.T.to_numpy()
    res = {}
    keys = ['price_mean', 'volume_pos', 'volume_neg', 'price_low', 'price_high', 'volume_total']
    for i in range(len(keys)):
        res[keys[i]] = temp[i]
    return res

def get_significance_levels(np_positions : np.array) -> Tuple:
    """
    Volume Profile is understood as grouping of volume into equidistant
    price bins for a certain period. The Value Area in a Profile is defined
    as those neighboring price bins that contain a certain precentage of
    volume traded in the whole period (e.g., 70%).
    This functions determines the following aspects of the Volume Profile
    and the contained Value Area:
    - Profile Low: the lowest price bin of profile
    - Profile High: the highest price bin of profile
    - Point of Control: the price bin with highest volume
    - Value Area Low: the lowest price bin of value area
    - Value Area High: the highest price bin of value area

    Args:
        np_positions(np.array): np.array containing the position of price bins 
                                in value area. Price bins not part of value area
                                contain np.nan.
    Returns:
        Features generated in form of list of bool np.arrays and a Dict
        that defines position of aspects defined above in the list.
    Raises:
        ValueError if arguments are invalid 
    """
    #Validate arguments
    if np_positions is None: raise ValueError('np_positions must not be None.')

    #Set up
    #Define keys and positions
    keys = {
        'profile_low' : 0,
        'profile_high' : 1,
        'point_of_control' : 2,
        'value_area_low' : 3,
        'value_area_high' : 4
    }
    #Preapre return arrays
    np_sig_levels = [np.zeros(np_positions.shape, dtype=bool) for i in range(len(keys))]
    
    #Calculation
    np_sig_levels[keys['profile_low']][0] = 1 #first bin
    np_sig_levels[keys['profile_high']][np_positions.shape[0]-1] = 1 #last bin
    np_sig_levels[keys['point_of_control']][np.where(np_positions == 1)[0][0]] = 1 #first set bin

    #first not nan bin
    for i in range(0, np_positions.shape[0]):
        if not np.isnan(np_positions[i]):
            np_sig_levels[keys['value_area_low']][i] = 1
            break
    
    #last not nan bin
    for i in range(np_positions.shape[0] - 1, 0, -1):
        if not np.isnan(np_positions[i]):
            np_sig_levels[keys['value_area_high']][i] = 1
            break

    return np_sig_levels, keys

def vp_extended(
    df : pd.DataFrame, 
    percent : float = 0.7,
    nr_bins : int = 10, 
    nr_volumes : int = 2,
    price_col = 'close') -> pd.DataFrame:

    """
    Creates a Volume Profile for price and volume data in passed DataFrame.
    Volume Profile is compised of a certain number of equidistant price bins
    (nr_bins) with aggreated volume (nr_volumes). A certain percentage (percent)
    of the Volume Profile constitutes the Value Area.
    Key aspects of the Volume Profile and the Value Area are:


    This function returns a DataFrame comprising Volume Profile and Value Area.
    The following features are included:
        price_low: Lower boundary of price bin (not including)
        price_mean: Average price in price bin
        price_high: Upper boundary of price bin (including)
        volume_total: Total volume in price bin (sum of negative and positive volume)
        volume_neg: Total negative volume in price bin (if current price < price of previous period)
        volume_pos: Total positive volume in price bin (if current price > price of previous period)
        position: Position of price bin in Value Area (1: Point of Control, 2-n: iteratively included price bins)
        profile_low: 1 if price bin is lowest in Volume Profile, otherwise 0.
        profile_high: 1 if price bin is highest in Volume Profile, otherwise 0.
        point_of_control: 1 if price bin is the one with te highest total volume, otherwise 0.
        value_area_low: 1 if price bin is the lowest in Value Area, otherwise 0.
        value_area_high: 1 if price bin is the highest in Value Area, otherwise 0.

    Arguments:
        df(pd.DataFrame): DtaFrame containing price (see price_col) and volume column
        percent(float): Percentage of volume representing Value Area (default: 0.7)
        nr_bins(int): Number of equidistant bins created for the Volume Profile (default: 10)
        nr_volumes(int): Number of volumes to aggreagte to create Value Area (default: 2)
        price_col(str): Name of price column in DataFrame (default: close)
    Raises:
        ValueError if arguments are invalid
    Returns:
        Features generated as specified above.
    """

    #Validate arguments
    if percent == 0.0 or percent > 1: raise ValueError('Percent must be great than 0.0 and be less or equal to 1.0')
    if nr_bins == 0: raise ValueError('Number of bins must not be 0.')
    if nr_volumes == 0 or nr_volumes > nr_bins - 1: raise ValueError('Bins to aggreate must be greater than 0 and smaller than nr_bins.')
    if not price_col in df.columns or not 'volume' in df.columns: raise ValueError('Column {0} or volume not found in DataFrame.'.format(price_col))

    #Setup
    va_volume = df.volume.sum() * percent
    np_cols = set_up(df=df,nr_bins=nr_bins, price_col=price_col)
    np_volume = np_cols['volume_total']
    np_pos = np.empty_like(np_volume)
    np_pos[:] = np.nan

    #Identify Point of Control (POC) and set initial volume
    idx = np.argmax(np_volume)
    current_volume = np_volume[idx]
    np_pos[idx] = 1
    
    #Get two nieghbor bins for initial comparison
    l_idx, l_volume, l_boundary = get_price_volume(
                np_volume=np_volume, 
                start_index=idx, 
                lower_direction=True,
                np_pos=np_pos,
                nr_volumes=nr_volumes
                )
    u_idx, u_volume, u_boundary = get_price_volume(
                np_volume=np_volume, 
                start_index=idx, 
                lower_direction=False,
                np_pos=np_pos,
                nr_volumes=nr_volumes
                )
    #Compare and asigned positions in value area as long as target volume
    #is not exceeded.
    counter = 2
    while (current_volume < va_volume):
        if np.isnan(u_volume) or l_volume >= u_volume:
            current_volume += l_volume    
            for l_i in l_idx:
                np_pos[l_i] = counter
            counter += 1
            if not l_boundary:
                l_idx, l_volume, l_boundary = get_price_volume(
                np_volume=np_volume, 
                start_index=idx, 
                lower_direction=True,
                np_pos=np_pos,
                nr_volumes=nr_volumes
                )
            else:
                l_idx, l_volume = np.nan, np.nan
        else:
            current_volume += u_volume
            for u_i in u_idx:
                np_pos[u_i] = counter
            counter += 1
            if not u_boundary:
                u_idx, u_volume, u_boundary = get_price_volume(
                np_volume=np_volume, 
                start_index=idx, 
                lower_direction=False,
                np_pos=np_pos,
                nr_volumes=nr_volumes
                )
            else:
                u_idx, u_volume = np.nan, np.nan, np.nan
        
    vpdf = pd.DataFrame({
        'price_low' : np_cols['price_low'],
        'price_mean' : np_cols['price_mean'],
        'price_high' : np_cols['price_high'],
        'volume_total' : np_volume,
        'volume_neg' : np_cols['volume_neg'],
        'volume_pos' : np_cols['volume_pos'],
        'position' : np_pos
    })

    #Add significance levels to returned DataFrame
    np_sig_levels, keys = get_significance_levels(np_pos)
    for key, value in keys.items():
        vpdf[key] = np_sig_levels[value]
    
    return vpdf