import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

def plot_ratio_vol_prod(ts_key: str, df_ratio: pd.DataFrame) -> None:
    """Plot Ratio-Volume-Production values for a given
    timeseries key

    Args:
        ts_key (str): timeseries key
        df_ratio (pd.DataFrame): vol/production ratio dataframe
    """
    # Plot Parameters
    #ts_key = 'Provider_10-Plant_1'
    plant = ts_key.split('-')[1]
    x_axis = 'Timestamp'
    vol1_axis = 'Actual_Vol_[Tons]'
    vol2_axis = 'Expected_Vol_[Tons]'
    prod_axis = 'Production'
    ratio_axis = 'Vol/Prod_ratio_kg'

    # Plot Variables
    _df = df_ratio.query(f" ts_key == '{ts_key}'")
    x = _df[x_axis]
    vol1 = _df[vol1_axis]
    vol2 = _df[vol2_axis]
    prod = _df[prod_axis]
    ratio = _df[ratio_axis]

    # Create a figure
    plt.rc('text',usetex=False)
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

    line_colors = {vol1_axis: '#1f76b4', 
                vol2_axis: '#ff7e0e', 
                prod_axis: '#107824', 
                ratio_axis: '#b41f49',
                'background_color': '#f0f0f0',
                }

    # Create Volume Plot
    axs[0].plot(x, vol1, label=vol1_axis, color=line_colors[vol1_axis])
    axs[0].plot(x, vol2, label=vol2_axis, color=line_colors[vol2_axis])  
    axs[0].set_title(f'Incoming Volume {ts_key}')
    axs[0].legend()
    axs[0].tick_params(axis='x', rotation=80)
    axs[0].set_facecolor(line_colors['background_color']) 
    axs[0].grid(True)

    # Production Plot
    axs[1].plot(x, prod, label=prod_axis, color=line_colors[prod_axis])
    axs[1].set_title(f'Production Level {plant}')
    axs[1].tick_params(axis='x', rotation=80)
    axs[1].legend()
    axs[1].set_facecolor(line_colors['background_color']) 
    axs[1].grid(True)

    # Ratio Plot
    axs[2].plot(x, ratio, label=ratio_axis, color=line_colors[ratio_axis])
    axs[2].set_title(f'Volume/Production Ratio {ts_key}')
    axs[2].tick_params(axis='x', rotation=80)
    axs[2].legend()
    axs[2].set_facecolor(line_colors['background_color']) 
    axs[2].grid(True)

    # Add labels and adjust layout
    plt.tight_layout()
    plt.show()
    plt.close('all')

def plot_ratio_all_ts(df_ratio: pd.DataFrame, path: str) -> None:
    """Plot Vol/Prod Ratio for all timeseries

    Args:
        df_ratio (pd.DataFrame): vol/production ratio dataframe
    """

    with PdfPages(path) as pdf:
    
        x_axis = 'Timestamp'
        vol1_axis = 'Actual_Vol_[Tons]'
        vol2_axis = 'Expected_Vol_[Tons]'
        prod_axis = 'Production'
        ratio_axis = 'Vol/Prod_ratio_kg'
        light_gray = '#f0f0f0'
        
        line_colors = {vol1_axis: '#1f76b4', 
                        vol2_axis: '#ff7e0e', 
                        prod_axis: '#107824', 
                        ratio_axis: '#b41f49',
                        }

        for ts_key in df_ratio['ts_key'].unique(): #[:10]:
                
            _df = df_ratio.query(f" ts_key == '{ts_key}'")
            plant = ts_key.split('-')[1]
            x = _df[x_axis]
            vol1 = _df[vol1_axis]
            vol2 = _df[vol2_axis]
            prod = _df[prod_axis]
            ratio = _df[ratio_axis]

            # Create a figure
            plt.rc('text',usetex=False)
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

            # Create Volume Plot
            axs[0].plot(x, vol1, label=vol1_axis, color=line_colors[vol1_axis])
            axs[0].plot(x, vol2, label=vol2_axis, color=line_colors[vol2_axis])  
            axs[0].set_title(f'Incoming Volume {ts_key}')
            axs[0].legend()
            axs[0].tick_params(axis='x', rotation=80)
            axs[0].set_facecolor(light_gray) 
            axs[0].grid(True)

            # Production Plot
            axs[1].plot(x, prod, label=prod_axis, color=line_colors[prod_axis])
            axs[1].set_title(f'Production Level {plant}')
            axs[1].tick_params(axis='x', rotation=80)
            axs[1].legend()
            axs[1].set_facecolor(light_gray) 
            axs[1].grid(True)

            # Ratio Plot
            axs[2].plot(x, ratio, label=ratio_axis, color=line_colors[ratio_axis])
            axs[2].set_title(f'Volume/Production Ratio {ts_key}')
            axs[2].tick_params(axis='x', rotation=80)
            axs[2].legend()
            axs[2].set_facecolor(light_gray) 
            axs[2].grid(True)

            # Add labels and adjust layout
            plt.tight_layout()
                
            # saves the current figure into a pdf page
            pdf.savefig(fig) 
            plt.close(fig)
            
            del _df

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'Timeseries Vol-Prod-Ratio'
        d['Author'] = 'John Torres'
        d['CreationDate'] = datetime.today()

        plt.close('all')