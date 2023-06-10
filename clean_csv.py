import pandas 

# Read the csv file
df = pandas.read_csv("./Data/Train_Tagged_Titles.csv")

#drop all rows where the tag == "No Tag"
df = df[df.Tag != "No Tag"]

#drop the "Unnamed: 0" column
df = df.drop(columns=["Unnamed: 4"])

#drop all rows where the tag == "No Tag"
df = df[df.Tag != "No Tag"]

#drop all rows where the tag is null
df = df.dropna(subset=["Tag"])


print(df.head())

#save the dataframe to a new csv file
df.to_csv("./Data/Train_Tagged_Titles_Clean.csv", index=False)