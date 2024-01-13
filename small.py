import pandas as pd

# Assuming your DataFrame is named df
# Replace df with your actual DataFrame name
df = pd.read_csv("dataset.csv")

# Selecting only rows with specified user names
selected_users = ['Roger Kaid', 'Rob Polley']
filtered_df = df[df['user_name'].isin(selected_users)]

# Saving the filtered DataFrame to a new file named small.df
filtered_df.to_csv('small.df', index=False)
