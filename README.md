Abstract:
Finding a restaurant that aligns with one's taste preferences, particularly in unfamiliar cities, poses a challenging task for travelers. This study addresses a persistent query that confronts travelers exploring new destinations: where should we dine? Leveraging data from TripAdvisor and incorporating information from restaurants in 31 of Europe's largest cities, our system is designed to simplify the restaurant selection process for tourists. Our recommender system is intricately developed, taking into account pivotal factors such as cuisine type, location, price range, and rating. This approach ensures the provision of personalized restaurant recommendations tailored to the specific preferences of individual travelers. Conducting a thorough comparative analysis of collaborative filtering, content-based filtering, and hybrid models, we finalized our recommender system through a comprehensive evaluation. This ensures its capability to suggest the perfect restaurant based on the nuanced preferences of travelers. In essence, this research not only alleviates a significant pain point for tourists but also contributes a valuable tool to enhance their overall travel experience in European cities.

Dataset comes from Kaggle at https://www.kaggle.com/datasets/damienbeneschi/krakow-ta-restaurans-data-raw

Importing dataset
import pandas as pd

# load csv to dataframe
rest = pd.read_csv("TA_restaurants_curated.csv")
rest.head()
Unnamed: 0	Name	City	Cuisine Style	Ranking	Rating	Price Range	Number of Reviews	Reviews	URL_TA	ID_TA
0	0	Martine of Martine's Table	Amsterdam	['French', 'Dutch', 'European']	1.0	5.0	
−
$	136.0	[['Just like home', 'A Warm Welcome to Wintry ...	/Restaurant_Review-g188590-d11752080-Reviews-M...	d11752080
1	1	De Silveren Spiegel	Amsterdam	['Dutch', 'European', 'Vegetarian Friendly', '...	2.0	4.5	
812.0	[['Great food and staff', 'just perfect'], ['0...	/Restaurant_Review-g188590-d693419-Reviews-De_...	d693419
2	2	La Rive	Amsterdam	['Mediterranean', 'French', 'International', '...	3.0	4.5	
567.0	[['Satisfaction', 'Delicious old school restau...	/Restaurant_Review-g188590-d696959-Reviews-La_...	d696959
3	3	Vinkeles	Amsterdam	['French', 'European', 'International', 'Conte...	4.0	5.0	
564.0	[['True five star dinner', 'A superb evening o...	/Restaurant_Review-g188590-d1239229-Reviews-Vi...	d1239229
4	4	Librije's Zusje Amsterdam	Amsterdam	['Dutch', 'European', 'International', 'Vegeta...	5.0	4.5	
316.0	[['Best meal.... EVER', 'super food experience...	/Restaurant_Review-g188590-d6864170-Reviews-Li...	d6864170
rest.shape
(125527, 11)
rest.columns
Index(['Unnamed: 0', 'Name', 'City', 'Cuisine Style', 'Ranking', 'Rating',
       'Price Range', 'Number of Reviews', 'Reviews', 'URL_TA', 'ID_TA'],
      dtype='object')
Data Cleaning and Preprocessing
# create new column for restaurant ID
column_to_rename = rest.columns[0]
new_column_name = 'Restaurant_ID'

rest = rest.rename(columns={column_to_rename: new_column_name})
rest.dtypes
Restaurant_ID          int64
Name                  object
City                  object
Cuisine Style         object
Ranking              float64
Rating               float64
Price Range           object
Number of Reviews    float64
Reviews               object
URL_TA                object
ID_TA                 object
dtype: object
#checking for missing values
rest.isnull()
number_of_missing_values = rest.isnull().sum()
number_of_missing_values
Restaurant_ID            0
Name                     0
City                     0
Cuisine Style        31351
Ranking               9651
Rating                9630
Price Range          47855
Number of Reviews    17344
Reviews               9616
URL_TA                   0
ID_TA                    0
dtype: int64
# get rid of missing values
rest = rest.dropna()
rest.shape
(74225, 11)
This dataset was obtained from the Trip Advisor tourism website by scraping through reviews and restaurant listing pages. After dropping all rows with missing values, our cleaned dataset involved 74,225 different restaurants ranging across all of the 31 different cities.

# change price to numeric rather than dollar signs
def convert_price_range(price):
    if isinstance(price, str):
        if '-' in price:
            # Handle the dash-separated ranges
            min_range, max_range = price.split('-')
            min_dollars = min_range.count('$')
            max_dollars = max_range.count('$')
            average_dollars = (min_dollars + max_dollars) / 2  # Calculate the average
            return average_dollars
        else:
            # Handle single price entries without dashes
            return price.count('$')
    return None

rest['Numeric Price'] = rest['Price Range'].apply(convert_price_range)

print(rest[['Price Range', 'Numeric Price']])
       Price Range  Numeric Price
0         $$ - $$$            2.5
1             $$$$            4.0
2             $$$$            4.0
3             $$$$            4.0
4             $$$$            4.0
...            ...            ...
125423    $$ - $$$            2.5
125434    $$ - $$$            2.5
125435    $$ - $$$            2.5
125438        $$$$            4.0
125445    $$ - $$$            2.5

[74225 rows x 2 columns]
# get rid of unhelpful columns
columns_to_drop = ["Price Range", "URL_TA", "ID_TA","Reviews"]
rest = rest.drop(columns=columns_to_drop, axis=1)
import numpy as np
# create random user IDs to be able to use collaborative and hybrid filtering
np.random.seed(42)
rest['User_ID'] = np.random.randint(1, 20001, size=len(rest))
# check for user ID
rest.head(10)
Restaurant_ID	Name	City	Cuisine Style	Ranking	Rating	Number of Reviews	Numeric Price	User_ID
0	0	Martine of Martine's Table	Amsterdam	['French', 'Dutch', 'European']	1.0	5.0	136.0	2.5	15796
1	1	De Silveren Spiegel	Amsterdam	['Dutch', 'European', 'Vegetarian Friendly', '...	2.0	4.5	812.0	4.0	861
2	2	La Rive	Amsterdam	['Mediterranean', 'French', 'International', '...	3.0	4.5	567.0	4.0	5391
3	3	Vinkeles	Amsterdam	['French', 'European', 'International', 'Conte...	4.0	5.0	564.0	4.0	11965
4	4	Librije's Zusje Amsterdam	Amsterdam	['Dutch', 'European', 'International', 'Vegeta...	5.0	4.5	316.0	4.0	11285
5	5	Ciel Bleu Restaurant	Amsterdam	['Contemporary', 'International', 'Vegetarian ...	6.0	4.5	745.0	4.0	6266
6	6	Zaza's	Amsterdam	['French', 'International', 'Mediterranean', '...	7.0	4.5	1455.0	2.5	16851
7	7	Blue Pepper Restaurant And Candlelight Cruises	Amsterdam	['Asian', 'Indonesian', 'Vegetarian Friendly',...	8.0	4.5	675.0	4.0	4427
8	8	Teppanyaki Restaurant Sazanka	Amsterdam	['Japanese', 'Asian', 'Vegetarian Friendly', '...	9.0	4.5	923.0	4.0	14424
9	9	Rob Wigboldus Vishandel	Amsterdam	['Dutch', 'Seafood', 'Fast Food']	10.0	4.5	450.0	1.0	11364
# change data types
rest['Ranking'] = rest['Ranking'].astype(float)
rest['Rating'] = rest['Rating'].astype(float)
rest['Number of Reviews'] = rest['Number of Reviews'].astype(float)
rest['Numeric Price'] = rest['Numeric Price'].astype(float)
# check data types
rest.dtypes
Restaurant_ID          int64
Name                  object
City                  object
Cuisine Style         object
Ranking              float64
Rating               float64
Number of Reviews    float64
Numeric Price        float64
User_ID                int64
dtype: object
Content Based Recommender System
# features to make recommendations based on
features = ['City', 'Cuisine Style', 'Ranking', 'Rating', 'Numeric Price', 'Number of Reviews']
# combine features as string to make new column
def combined_features(row):
    return row['City']+ " "+row['Cuisine Style']+" " +str(row['Ranking'])+" "+str(row['Rating'])+"  "+str(row['Numeric Price'])+"  "+str(row['Number of Reviews'])
rest["combined_features"] = rest.apply(combined_features, axis =1)
rest.head()
Restaurant_ID	Name	City	Cuisine Style	Ranking	Rating	Number of Reviews	Numeric Price	User_ID	combined_features
0	0	Martine of Martine's Table	Amsterdam	['French', 'Dutch', 'European']	1.0	5.0	136.0	2.5	15796	Amsterdam ['French', 'Dutch', 'European'] 1.0 ...
1	1	De Silveren Spiegel	Amsterdam	['Dutch', 'European', 'Vegetarian Friendly', '...	2.0	4.5	812.0	4.0	861	Amsterdam ['Dutch', 'European', 'Vegetarian Fr...
2	2	La Rive	Amsterdam	['Mediterranean', 'French', 'International', '...	3.0	4.5	567.0	4.0	5391	Amsterdam ['Mediterranean', 'French', 'Interna...
3	3	Vinkeles	Amsterdam	['French', 'European', 'International', 'Conte...	4.0	5.0	564.0	4.0	11965	Amsterdam ['French', 'European', 'Internationa...
4	4	Librije's Zusje Amsterdam	Amsterdam	['Dutch', 'European', 'International', 'Vegeta...	5.0	4.5	316.0	4.0	11285	Amsterdam ['Dutch', 'European', 'International...
# check column
rest['combined_features'].head(10)
0    Amsterdam ['French', 'Dutch', 'European'] 1.0 ...
1    Amsterdam ['Dutch', 'European', 'Vegetarian Fr...
2    Amsterdam ['Mediterranean', 'French', 'Interna...
3    Amsterdam ['French', 'European', 'Internationa...
4    Amsterdam ['Dutch', 'European', 'International...
5    Amsterdam ['Contemporary', 'International', 'V...
6    Amsterdam ['French', 'International', 'Mediter...
7    Amsterdam ['Asian', 'Indonesian', 'Vegetarian ...
8    Amsterdam ['Japanese', 'Asian', 'Vegetarian Fr...
9    Amsterdam ['Dutch', 'Seafood', 'Fast Food'] 10...
Name: combined_features, dtype: object
# make vectors to do cosine similarity
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
count_matrix = count_vector.fit_transform(rest["combined_features"])
count_vector.get_feature_names_out()
array(['10', '100', '1000', ..., 'yunnan', 'zealand', 'zurich'],
      dtype=object)
count_vector.get_feature_names_out().shape
(13463,)
from sklearn.metrics.pairwise import cosine_similarity

# get name index in the dataframe for quick look-up
name_index = pd.Series(rest.index, index=rest['Name']).to_dict()
def get_index_from_name(name):
    return name_index[name]

def get_name_from_index(index):
    return rest['Name'].iloc[index]
def content_based_recommender(rest_name, num_of_rec=5):
    name_idx = get_index_from_name(rest_name)
    name_vec = count_vector.transform([rest['combined_features'].iloc[name_idx]])
    cosine_sim = cosine_similarity(name_vec, count_matrix)

    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommended_indices = []
    recommended_restaurants = set()
    for idx, score in sim_scores:
        if rest['Name'].iloc[idx] not in recommended_restaurants:
            recommended_indices.append(idx)
            recommended_restaurants.add(rest['Name'].iloc[idx])
            if len(recommended_indices) == num_of_rec + 1:
                break

    recommendations = [(rest['Name'].iloc[idx]) for idx in recommended_indices[1:]]
    return recommendations
# restaurant you visited
name_visited = "De Balie"
def get_cuisine_style_from_dataset(item_description):
    row = rest[rest['Name'] == item_description]
    if not row.empty:
        return row['Cuisine Style'].values[0]
    else:
        return "Unknown"
# final output for content based
recommended_items = content_based_recommender(name_visited, 5)

print(f" 5 similar restaurants you may like based on your interest in {name_visited}:")
print("**********************************************************")
for idx, (restaurant_name) in enumerate(recommended_items):
    print(f"{idx+1}. {restaurant_name}")
    cuisine_style = get_cuisine_style_from_dataset(restaurant_name)  # Replace this with your logic
    print(f"   - Cuisine style: {cuisine_style}")
print("************************************************************")
print("                   *Content Based Model*                    ")
 5 similar restaurants you may like based on your interest in De Balie:
**********************************************************
1. De Vondeltuin
   - Cuisine style: ['Dutch', 'Bar', 'International', 'European', 'Pub', 'Vegetarian Friendly']
2. Bar Mick
   - Cuisine style: ['Bar', 'European', 'Dutch', 'International', 'Pub']
3. Brasserie Blazer
   - Cuisine style: ['Bar', 'European', 'Pub', 'Dutch', 'International']
4. Rooster Amsterdam
   - Cuisine style: ['Dutch', 'Bar', 'European', 'Pub', 'International']
5. Verhulst
   - Cuisine style: ['Dutch', 'European', 'Bar', 'International', 'Pub']
************************************************************
                   *Content Based Model*                    
Collaborative Based Filtering
#user-item interaction matrix
user_item_matrix = rest.pivot_table(index='User_ID',columns='Name',values='Rating',aggfunc='sum',fill_value=0)
#  basic info + few rows of matrix
(user_item_matrix.info(), user_item_matrix.head())
<class 'pandas.core.frame.DataFrame'>
Int64Index: 19515 entries, 1 to 20000
Columns: 67862 entries, "52" Bistro Restauarnt and Bar to 美心酒家（Mei Xin Restaurant）
dtypes: float64(35183), int64(32679)
memory usage: 9.9 GB
(None,
 Name     "52" Bistro Restauarnt and Bar  "Above" Roof Top Restaurant  \
 User_ID                                                                
 1                                     0                            0   
 2                                     0                            0   
 3                                     0                            0   
 4                                     0                            0   
 5                                     0                            0   
 
 Name     "Bistro Antidotum"  "Kepzeld el!" Wine bar  \
 User_ID                                               
 1                         0                     0.0   
 2                         0                     0.0   
 3                         0                     0.0   
 4                         0                     0.0   
 5                         0                     0.0   
 
 Name     "SPECIAL" Hamburger & Italian Fast Food  #Citypie - Fast Slow Food  \
 User_ID                                                                       
 1                                            0.0                        0.0   
 2                                            0.0                        0.0   
 3                                            0.0                        0.0   
 4                                            0.0                        0.0   
 5                                            0.0                        0.0   
 
 Name     #Hashtag  #Marionett Craft Beer House  &Samhoud Places  'A Pazziella  \
 User_ID                                                                         
 1               0                          0.0              0.0           0.0   
 2               0                          0.0              0.0           0.0   
 3               0                          0.0              0.0           0.0   
 4               0                          0.0              0.0           0.0   
 5               0                          0.0              0.0           0.0   
 
 Name     ...  Αmandine  Εl Τaco Βueno  Εxou  Η Παληα Αθηνα  Ι Folia Chrysina  \
 User_ID  ...                                                                   
 1        ...       0.0              0     0            0.0               0.0   
 2        ...       0.0              0     0            0.0               0.0   
 3        ...       0.0              0     0            0.0               0.0   
 4        ...       0.0              0     0            0.0               0.0   
 5        ...       0.0              0     0            0.0               0.0   
 
 Name     Τhe Greco’s Sea Prj Monastiraki  Τα Φιλαρακια  \
 User_ID                                                  
 1                                    0.0             0   
 2                                    0.0             0   
 3                                    0.0             0   
 4                                    0.0             0   
 5                                    0.0             0   
 
 Name     ​Byward Kitchen and Bar  ​La Valona  美心酒家（Mei Xin Restaurant）  
 User_ID                                                                 
 1                            0.0           0                       0.0  
 2                            0.0           0                       0.0  
 3                            0.0           0                       0.0  
 4                            0.0           0                       0.0  
 5                            0.0           0                       0.0  
 
 [5 rows x 67862 columns])
user_item_matrix.shape
(19515, 67862)
SVD
# imports
from scipy.sparse.linalg import svds
# make data type float
user_item_matrix_float = user_item_matrix.astype('float')
# Singular Value Decomposition
U, sigma, Vt = svds(user_item_matrix_float.values, k=50)  # k is the number of latent factors
# sigma to a diagonal matrix
sigma_diag_matrix = np.diag(sigma)
# predicted ratings
predicted_ratings = np.dot(np.dot(U, sigma_diag_matrix), Vt)

#  shape of the predicted ratings (to compare to og make sure have all features)#hooray
predicted_ratings.shape, user_item_matrix.shape
((19515, 67862), (19515, 67862))
# Create a DataFrame for predicted ratings
predicted_ratings_df = pd.DataFrame(predicted_ratings,
                                    columns=user_item_matrix.columns,
                                    index=user_item_matrix.index)
def svd_collab_rec(user_id, num_recommendations=5):
    # check if user_id is in the user-item matrix
    if user_id not in user_item_matrix.index:
        raise ValueError("User ID not found in the user-item matrix.")

    # getting the index of user_id in the user-item matrix
    user_idx = user_item_matrix.index.get_loc(user_id)

    # predicted ratings for the user
    user_predicted_ratings = predicted_ratings[user_idx, :]

    # Id items that the user has not interacted with
    user_actual_ratings = user_item_matrix.loc[user_id, :]
    unrated_items_mask = user_actual_ratings == 0
    unrated_items_idx = np.where(unrated_items_mask)[0]

    # predicted ratings for unrated items and id top-rated items
    unrated_predicted_ratings = user_predicted_ratings[unrated_items_idx]
    top_rated_items_idx = unrated_items_idx[np.argsort(-unrated_predicted_ratings)[:num_recommendations]]

    # descriptions of top-rated items
    recommended_items = user_item_matrix.columns[top_rated_items_idx].tolist()

    return recommended_items
# get recommendations for specified user
svd_collab_rec(1234, num_recommendations=5)
['Sunny',
 'Khartoum Cafe',
 'MnM Gelato',
 'da Roberto e Loretta',
 'Le Petit Boileau']
print(rest[rest['User_ID'] == 1234])
       Restaurant_ID           Name       City  \
32168            225   Red Squirrel  Edinburgh   
44563           3470        O Lirio     Lisbon   
53434           7854  Pret a Manger     London   

                                           Cuisine Style  Ranking  Rating  \
32168  ['Bar', 'British', 'Pub', 'Scottish', 'Vegetar...    226.0     4.0   
44563        ['Mediterranean', 'European', 'Portuguese']   3474.0     3.0   
53434                                        ['British']   7863.0     4.0   

       Number of Reviews  Numeric Price  User_ID  \
32168              584.0            2.5     1234   
44563              120.0            1.0     1234   
53434               13.0            1.0     1234   

                                       combined_features  
32168  Edinburgh ['Bar', 'British', 'Pub', 'Scottish'...  
44563  Lisbon ['Mediterranean', 'European', 'Portugue...  
53434           London ['British'] 7863.0 4.0  1.0  13.0  
# final output of collaborative system 
userid = 1234
recommended_items = svd_collab_rec(userid, 5)

print(f"Recommended restaurants for User #{userid}:")
print("**********************************************************")
for i, item_description in enumerate(recommended_items, 1):
    print(f"{i}. {item_description}")
    cuisine_style = get_cuisine_style_from_dataset(item_description)  
    print(f"   - Cuisine style: {cuisine_style}")
print("************************************************************")
print("                  *Collaborative Model*                     ")
Recommended restaurants for User #1234:
**********************************************************
1. Sunny
   - Cuisine style: ['Chinese', 'Japanese', 'Asian', 'Thai']
2. Khartoum Cafe
   - Cuisine style: ['Cafe', 'Fast Food', 'Middle Eastern', 'Vegetarian Friendly', 'Vegan Options']
3. MnM Gelato
   - Cuisine style: ['Cafe']
4. da Roberto e Loretta
   - Cuisine style: ['Italian', 'Mediterranean', 'Vegetarian Friendly', 'Gluten Free Options']
5. Le Petit Boileau
   - Cuisine style: ['French', 'European', 'Gastropub']
************************************************************
                  *Collaborative Model*                     
Hybrid Model
def hybrid_recommendation(user_id, rest_name, num_recommendations=5):
    # Get recommendations from content-based model
    content_recommendations = content_based_recommender(rest_name, num_recommendations*2)

    # Get recommendations from collaborative model
    collaborative_recommendations = svd_collab_rec(user_id, num_recommendations*2)

    # combine recs and make sure no duplicates
    combined_recommendations = list(set(content_recommendations + collaborative_recommendations))

    # if too many recs - choose the top N
    if len(combined_recommendations) > num_recommendations:
        combined_recommendations = combined_recommendations[:num_recommendations]

    print(f"\nRecommended items for User #{user_id}, based on visiting {rest_name}:")
    print("**********************************************************")
    for i, reco_item_description in enumerate(combined_recommendations, 1):
    # Assuming get_cuisine_style_from_dataset() retrieves cuisine style from your dataset based on reco_item_description
        cuisine_style = get_cuisine_style_from_dataset(reco_item_description)
        print(f"{i}. {reco_item_description}")
        print(f"   - Cuisine type: {cuisine_style}")
    print("************************************************************")
def get_recommendations(rest, sim_scores, num_of_rec):
    recommended_indices = []
    recommended_restaurants = set()
    
    for idx, score in sim_scores:
        if rest['Name'].iloc[idx] not in recommended_restaurants:
            recommended_indices.append(idx)
            recommended_restaurants.add(rest['Name'].iloc[idx])
            if len(recommended_indices) == num_of_rec + 1:
                break
    
    return recommended_indices
def hybrid_recommendation(user_id, rest_name, k=10):

  # k: number of recommended restaurants

  # get index from the title
    rest_index=get_index_from_name(rest_name)

  # pairwise cosine value for given restaurant index
    name_idx = get_index_from_name(rest_name)
    name_vec = count_vector.transform([rest['combined_features'].iloc[name_idx]])
    cosine_sim = cosine_similarity(name_vec, count_matrix)

    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Store indices for restaurants with high cosine value
    recommended_indices = get_recommendations(rest, sim_scores, k)

  # get restaurant id from recommended_indices
    name_ids=[get_name_from_index(i) for i in recommended_indices]

  # Get predicted value of the restaurant for the user
    predicted_rating= [predicted_ratings_df.loc[user_id][i] for i in name_ids]

  # making a dataframe with restaurant_ids with predicted value
    rest_rating=pd.DataFrame({'Restaurant_ID': name_ids, 'PredRating': predicted_rating})

  # Sort the DataFrame based on 'pred_ratings'
    sorted_name_rating = rest_rating.sort_values(by='PredRating',ascending=False)

  # Getting list of top k restaurant's restaurant id
    top_k_name_ids= sorted_name_rating['Restaurant_ID'].values[0:k].tolist()

    print(f"\nRecommended restaurants for User #{user_id} after visiting {rest_name} are:")
    print("**********************************************************")
    for i, name_id in enumerate(top_k_name_ids, 1):
        movie_title = rest[rest['Restaurant_ID'] == name_id]['Name'].values
        cuisine_style = get_cuisine_style_from_dataset(name_id)  # Assuming you have a function to fetch cuisine style from the ID
        print(f"{i}. {name_id}")
        # Adjust the 'movie_title' print as per the desired format
        if len(movie_title) > 0:
            print(f"   - Title: {movie_title[0]}")
        print(f"   - Cuisine Style: {cuisine_style}")  # Print the cuisine style
    print("**********************************************************")
    print("                  *Hybrid Model*                     ")
# final output of hybrid filtering model
user_id = 1234  # Example restaurant ID
rest_name = "De Balie"  # restaurant name 
num_recommendations = 5  # Number of recs

#Hybrid recs
hybrid_recommendation(user_id, rest_name, num_recommendations)
Recommended restaurants for User #1234 after visiting De Balie are:
**********************************************************
1. Brasserie Blazer
   - Cuisine Style: ['Bar', 'European', 'Pub', 'Dutch', 'International']
2. Kaap de Goede Hoop
   - Cuisine Style: ['German', 'Dutch', 'European', 'Bar', 'International', 'Pub']
3. Bar Mick
   - Cuisine Style: ['Bar', 'European', 'Dutch', 'International', 'Pub']
4. Rooster Amsterdam
   - Cuisine Style: ['Dutch', 'Bar', 'European', 'Pub', 'International']
5. Verhulst
   - Cuisine Style: ['Dutch', 'European', 'Bar', 'International', 'Pub']
**********************************************************
                  *Hybrid Model*                     
Pros and Cons about each filtering method
Content Based Filtering System: Pros: -Accurately recommended similar resturants to each resturant name it was provided Cons: -The restaurant that someone visited is coded into the model itself and cannot be changed with ease

Collaborative Filtering Sytem: Pros: -Accurately recommended resturants that the user may like Cons: -Data may not have been accurate. To simulate different user ids, we generated random numbers from 1-20000 and assigned them to random restarants and the data that went along with them. This caused the system to give inaccurate results because the range of restaurants the user visited varied greatly due to its random state.

Hybrid Filtering System: Pros: -Accurately took data from both content and collaborative systems and presented correctly to user Cons: -Because of the random nature of the collaboritive model, the hybrid model also had some inaccurate recommendations

After reviewing and comparing each of the models, we decided that the content based model gave the most accurate results. The inaccuracy of the other models was due to the random nature of the collaborative model. In the future, we hope to find a dataset that includes the id of the users that rated each restaurant to more accurately recommend similar restaurants.
