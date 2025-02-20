import os
# import pandas as pd
import matplotlib.pyplot as plt
import csv
import json

def plot_metrics():

    log_dir = os.getenv("OPENAI_LOGDIR")
    # Path to the CSV log file
    # csv_file = os.path.join(log_dir, 'progress.csv')
    #
    # # Check if the CSV file exists
    # if not os.path.exists(csv_file):
    #     print(f"CSV log file not found at {csv_file}")
    #     return
    #
    # # Read the CSV file into a DataFrame
    # df = pd.read_csv(csv_file)
    #
    # # Convert 'step' or 'epoch' column to integer if necessary
    # if 'step' in df.columns:
    #     df['epoch'] = df['step']
    # elif 'epoch' in df.columns:
    #     df['epoch'] = df['epoch']
    # else:
    #     print("No 'step' or 'epoch' column found in the CSV file.")
    #     return
    #
    # # Metrics to plot
    # metrics = [
    #     'current_grad_norm',
    #     'current_param_norm',
    #     'grad_norm',
    #     'loss',
    #     'loss_q0',
    #     'loss_q1',
    #     'loss_q2',
    #     'loss_q3',
    #     'mse_image',
    #     'mse_image_q0',
    #     'mse_image_q1',
    #     'mse_image_q2',
    #     'mse_image_q3',
    #     'mse_tabular',
    #     'mse_tabular_q0',
    #     'mse_tabular_q1',
    #     'mse_tabular_q2',
    #     'mse_tabular_q3',
    #     'param_norm',
    # ]
    #
    # eval_metrics = ['FID Score', 'MMD Score', 'Tabular MMD Score']
    #
    # # Create a figure and axis for the first graph
    # fig1, ax1 = plt.subplots(figsize=(12, 8))
    #
    # # Plot each metric
    # for metric in metrics:
    #     if metric in df.columns:
    #         ax1.plot(df['epoch'], df[metric], label=metric)
    #     else:
    #         print(f"Metric '{metric}' not found in CSV file.")
    #
    # # Customize the first plot
    # ax1.set_title('Training Metrics Over Epochs')
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Value')
    # ax1.legend(loc='upper right')
    # ax1.grid(True)
    #
    # # Save the first plot
    # plt.tight_layout()
    # fig1.savefig(os.path.join(log_dir, 'training_metrics.png'))
    # print(f"Training metrics plot saved to {os.path.join(log_dir, 'training_metrics.png')}")
    #
    # # Create a figure and axis for the second graph
    # fig2, ax2 = plt.subplots(figsize=(12, 8))
    #
    # # Plot evaluation metrics
    # for metric in eval_metrics:
    #     if metric in df.columns:
    #         ax2.plot(df['epoch'], df[metric], label=metric)
    #     else:
    #         print(f"Evaluation metric '{metric}' not found in CSV file.")
    #
    # # Customize the second plot
    # ax2.set_title('Evaluation Metrics Over Epochs')
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Value')
    # ax2.legend(loc='upper right')
    # ax2.grid(True)
    #
    # # Save the second plot
    # plt.tight_layout()
    # fig2.savefig(os.path.join(log_dir, 'evaluation_metrics.png'))
    # print(f"Evaluation metrics plot saved to {os.path.join(log_dir, 'evaluation_metrics.png')}")
    #
    # # Optionally, display the plots
    # # plt.show()