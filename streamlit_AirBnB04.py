# -*- coding: utf-8 -*-

import streamlit as st
from streamlit_folium import folium_static
import folium
import geocoder
from geopy.distance import geodesic
import pickle
import numpy as np   
import pandas as pd
import xgboost as xgb


from folium.plugins import HeatMap , MarkerCluster
from folium import plugins



import warnings
warnings.filterwarnings('ignore')

# Dictionnaire des coordonnées des centre-ville des 39 villes du dataset
geo_main_cities = {'Amsterdam':(52.37029683475726, 4.900181800886505), 
                   'Antwerpen':(51.21806636352488, 4.409214583292194), 
                   'Athina':(37.9874990005841, 23.73028106197067), 
                   'Austin':(30.287956448355878, -97.7403968572119), 
                   'Barcelona':(41.3894629923458, 2.169370263652049),
                   'Berlin':(52.53888485809899, 13.378196165332007), 
                   'Boston':(42.36246216247953, -71.04508424559839), 
                   'Bruxelles':(50.84436814560585, 4.362510502369775), 
                   'Chicago':(41.895984188093635, -87.63017389656089), 
                   'Denver':(39.74155350099731, -104.96018945390583), 
                   'Dublin':(53.34962656243507, -6.2882235229811085), 
                   'Edinburgh':(55.949465918487256, -3.199841261060984), 
                   'Genève':(46.20359169297537, 6.148550826168727), 
                   'Hong Kong':(22.3206296986255, 114.17180946213102), 
                   'København':(55.68943629649777, 12.575994324928013), 
                   'London':(51.496392311426675, -0.11552617375276861), 
                   'Los Angeles':(34.05214470040869, -118.28217054909794), 
                   'Madrid':(40.41688927444238, -3.68782006777526), 
                   'Mallorca':(39.57104151669509, 2.6547087709852892), 
                   'Manchester':(53.491381823443525, -2.2453121925528605), 
                   'Melbourne':(-37.80134703712602, 144.9696126652144), 
                   'Montréal':(45.505020341425855, -73.57130244909894), 
                   'Nashville':(36.83800623466486, -86.79372364425053), 
                   'New Orleans':(30.00523472329412, -90.05842481373173), 
                   'New York':(40.76731726898433, -74.02768271377673), 
                   'Oakland':(37.80448420412626, -122.27355852700386), 
                   'Paris':(48.860570654452985, 2.3381084537380787), 
                   'Portland':(45.817480513937575, -122.69083959566777), 
                   'Québec':(46.88723452144307, -71.21405502378049), 
                   'Roma':(41.88695093084086, 12.491324637796852), 
                   'San Diego':(32.71549507835408, -117.17791223573323), 
                   'San Francisco':(37.84982202128698, -122.42448954564261), 
                   'Seattle':(47.65940435070754, -122.32969528649929), 
                   'Sydney':(-33.80250920917693, 151.18493647684207), 
                   'Toronto':(43.65875141920644, -79.35106522112997), 
                   'Vancouver':(49.37623687508625, -123.12795714425819), 
                   'Venezia':(45.48844056928191, 12.31223591815801), 
                   'Washington':(38.96048110529176, -76.969037646025), 
                   'Wien':(48.209730515986315, 16.399254432310666)}

# création du df data_villes
df_villes = pd.DataFrame(data=geo_main_cities).T.reset_index()
df_villes.columns = ['Ville','lat','long']
df_villes['geodesic'] = df_villes.apply(lambda row : (row['lat'], row['long']), axis=1)

# Chargement du dataset ayant servi à la modélisation
df = pd.read_csv("airbnb-clean_v2_mod_streamlit.csv",sep  = ';')
df = df.drop_duplicates()


# Liste des villes
#
#ville_list = sorted(df['Ville'].unique().tolist())
ville_list = ['Paris']+sorted(df['Ville'].unique().tolist())
#ville_list = ("Paris","London","New York","Madrid")


st.title("Prédiction du prix de nuitée d'un logement AirBnB")
st.sidebar.subheader("Localisation du bien")

ville_cour = st.sidebar.selectbox(label = "Choisir la ville", options = ville_list)

quartier_list = sorted(df[df['Ville']==ville_cour]['Quartier'].unique().tolist())
quartier_cour = st.sidebar.selectbox(label = "Sélectionner le quartier", options = quartier_list)

adresse = quartier_cour+" "+ville_cour
dist_centr_cour = 0
#st.write("adresse :",adresse)

adresse = adresse+" "+quartier_cour+" "+ville_cour

g = geocoder.osm(adresse)
#st.write(g.osm)
adresse1 = ""
try:
	#adresse = adresse+" "+quartier_cour+" "+ville_cour
	adresse = st.text_input('Adresse (format N°, Nom de rue)', value=''+g.osm["addr:street"]+', '+ville_cour+', '+g.osm["addr:country"])
	adresse1 = adresse
	g = geocoder.osm(adresse)
	m = folium.Map(location=[g.osm["y"], g.osm["x"]], zoom_start=16)
	tooltip = "MON BIEN",
	folium.Marker([g.osm["y"], g.osm["x"]], popup ="C'est ICI", tooltip=tooltip,icon = folium.Icon(icon='home', prefix="fa", color='red')).add_to(m)
	st.sidebar.write("**Adresse retenue** :")

	df_selec = df[(df.Quartier==quartier_cour)]
	df_selec['distance_au_bien'] = df_selec.apply(lambda row : geodesic((g.osm["y"], g.osm["x"]), (row['Latitude'], row['Longitude'])).km, axis=1)
	df_selec = df_selec.sort_values(by = 'distance_au_bien').head(5)

	#st.write("df :",df_selec.shape[0])
	if df_selec.shape[0] != 0:
    		for i in range(df_selec.shape[0]):
        		tooltip = "PRIX "+str(int(df_selec.iloc[i,10]))+" € - clic pour + infos"
        		popup_inside = ("{price} €<br>"
                	        "{nb_invites} personnes<br>"
                        	"{nb_chambres} chambres<br>"
                        	"{nb_lits} lits<br>"
                        	"{nb_sdb} sdb<br>"
                        	"{t_logt}<br>"
                        	"{p_logt}<br>"
                		).format(price=str(int(df_selec.iloc[i,10])),
                         		nb_invites=str(df_selec.iloc[i,11]),
                         		nb_chambres=str(df_selec.iloc[i,7]),
                         		nb_lits=str(df_selec.iloc[i,8]),
                         		nb_sdb=str(df_selec.iloc[i,6]),
                         		t_logt=str(df_selec.iloc[i,4]),
                         		p_logt=str(df_selec.iloc[i,5]),
                        	)
        		folium.Marker([df_selec.iloc[i,2], df_selec.iloc[i,3]], 
                      		popup=folium.Popup(popup_inside, max_width=150, min_width=30),
                      		tooltip=tooltip).add_to(m)
	folium_static(m, width=800, height=400)

except Exception as inst:
	adresse = ""
	adresse = adresse+" "+quartier_cour+" "+ville_cour
	g = geocoder.osm(adresse)
	#st.write("adresse1 :",adresse)
	#st.write("adresse2 :",adresse)
	if adresse == adresse1 or adresse1 == "" :
		adresse = st.text_input('Adresse (format N°, Nom de rue)', value="",placeholder =adresse)
	#st.write("adresse2 :",adresse)
	if adresse != "":
		g = geocoder.osm(adresse)
	
	m = folium.Map(location=[g.osm["y"], g.osm["x"]], zoom_start=16)
	tooltip = "MON BIEN"
	folium.Marker([g.osm["y"], g.osm["x"]], popup="C'est ICI", tooltip=tooltip,icon = folium.Icon(icon='home', prefix="fa", color='red')).add_to(m)
	st.sidebar.write("**Adresse retenue** :")

	df_selec = df[(df.Quartier==quartier_cour)]
	df_selec['distance_au_bien'] = df_selec.apply(lambda row : geodesic((g.osm["y"], g.osm["x"]), (row['Latitude'], row['Longitude'])).km, axis=1)
	df_selec = df_selec.sort_values(by = 'distance_au_bien').head(5)

	if df_selec.shape[0] != 0:
    		for i in range(df_selec.shape[0]):
        		tooltip = "PRIX "+str(int(df_selec.iloc[i,10]))+" € - clic pour + infos"
        		popup_inside = ("{price} €<br>"
                	        "{nb_invites} personnes<br>"
                        	"{nb_chambres} chambres<br>"
                        	"{nb_lits} lits<br>"
                        	"{nb_sdb} sdb<br>"
                        	"{t_logt}<br>"
                        	"{p_logt}<br>"
                		).format(price=str(int(df_selec.iloc[i,10])),
                         		nb_invites=str(df_selec.iloc[i,11]),
                         		nb_chambres=str(df_selec.iloc[i,7]),
                         		nb_lits=str(df_selec.iloc[i,8]),
                         		nb_sdb=str(df_selec.iloc[i,6]),
                         		t_logt=str(df_selec.iloc[i,4]),
                         		p_logt=str(df_selec.iloc[i,5]),
                        	)
        		folium.Marker([df_selec.iloc[i,2], df_selec.iloc[i,3]], 
                      		popup=folium.Popup(popup_inside, max_width=150, min_width=30),
                      		tooltip=tooltip).add_to(m)
	folium_static(m, width=800, height=400)


g = geocoder.osm(adresse)

# recherche dans df du prix moyen de quartier_cour
prix_moyen_cour = df[df['Quartier'] == quartier_cour]['Prix_moyen'].mean()

try:
	st.sidebar.write(g.osm["addr:housenumber"]+' '+g.osm["addr:street"]+', '+ville_cour+', '+g.osm["addr:country"])
	st.sidebar.subheader("Caractéristiques du bien")
	# calcul distance du bien au centre-ville
	dist_centr_cour = geodesic((g.osm["y"], g.osm["x"]),df_villes[df_villes.Ville==ville_cour]['geodesic']).km
	st.write("Distance au centre : "+str(round(dist_centr_cour,3))+" km"+"________________________________________Prix moyen quartier : "+str(round(prix_moyen_cour,2)))
except Exception as inst:
	print("except :",inst)


Type_logt_list = ('Apartment', 'House','Others')
Type_logt = st.sidebar.radio(label = "Type de logement", options = Type_logt_list)

Partage_logt_list =('Entire home/apt', 'Private room', 'Shared room')
Partage_logt = st.sidebar.radio(label = "Partage du logement", options = Partage_logt_list )

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
nb_invites_inclus = st.sidebar.radio("Nombre d'invité(s) max :", ("1","2","3","4","5","6","7"))

Nb_chambres_list =("1","2","3","4","5","6")
Nb_chambres = st.sidebar.radio(label = "Nombre de chambre(s)", options = Nb_chambres_list)

Nb_sdb_list = ("1","2","3")
Nb_sdb = st.sidebar.radio(label = "Nombre de sdb", options = Nb_sdb_list)

Nb_lits_list = ("1","2","3","4","5","6")
Nb_lits = st.sidebar.radio(label = "Nombre de lit(s)", options = Nb_lits_list)

Type_lit_list = ('Real Bed', 'Pull-out Sofa', 'Couch', 'Futon', 'Airbed')
Type_lit = st.sidebar.selectbox(label = "Type de lit", options = Type_lit_list)

Equipements_list = ('24-hour check-in', 'self check-in', 'iron','air conditioning', 'smoke detector', 'carbon monoxide detector',
       		    'cable tv', 'dryer', 'family/kid friendly', 'tv')
Equipements = st.sidebar.multiselect('Equipements éventuels', Equipements_list )

st.sidebar.subheader("Conditions contractuelles")

Nb_min_nuits_list = ("1","2","3","4","5","6")
Nb_min_nuits = st.sidebar.radio(label = "Nombre minimum de nuit(s)", options = Nb_min_nuits_list)

Annulation_pol_list = ('flexible', 'moderate', 'strict')
Annulation_pol = st.sidebar.radio(label = "Politique d'annulation", options = Annulation_pol_list)

adresse = adresse+" "+quartier_cour
g = geocoder.osm(adresse)


XGB_global = xgb.Booster({'nthread': 4}) 
XGB_global.load_model('airbnb_global_reg_.xbmodel')


q = 'Quartier_'+quartier_cour
#st.write(q)
#st.write(xg_Paris)

e = Equipements
i = 'Nb_invites_inclus'
ch = 'Nb_chambres_'+Nb_chambres
sdb = 'Nb_sdb_'+Nb_sdb
lits = 'Nb_lits_'+Nb_lits
nuits = 'Nb_min_nuits'
tlits = 'Type_lit_'+Type_lit
part = 'Partage_logt_'+Partage_logt
tlogt = 'Type_logt_'+Type_logt 
flex = 'Annulation_pol_'+Annulation_pol
pm = 'Prix_moyen'
dc = 'dist_centr'

list_col_train = [
 'Annulation_pol_flexible',
 'Annulation_pol_moderate',
 'Annulation_pol_strict',
 'Type_lit_Airbed',
 'Type_lit_Couch',
 'Type_lit_Futon',
 'Type_lit_Pull-out Sofa',
 'Type_lit_Real Bed',
 'Partage_logt_Entire home/apt',
 'Partage_logt_Private room',
 'Partage_logt_Shared room',
 'Type_logt_Apartment',
 'Type_logt_House',
 'Type_logt_Others',
 'Ville_Amsterdam',
 'Ville_Antwerpen',
 'Ville_Athina',
 'Ville_Austin',
 'Ville_Barcelona',
 'Ville_Berlin',
 'Ville_Boston',
 'Ville_Bruxelles',
 'Ville_Chicago',
 'Ville_Denver',
 'Ville_Dublin',
 'Ville_Edinburgh',
 'Ville_Genève',
 'Ville_Hong Kong',
 'Ville_København',
 'Ville_London',
 'Ville_Los Angeles',
 'Ville_Madrid',
 'Ville_Mallorca',
 'Ville_Manchester',
 'Ville_Melbourne',
 'Ville_Montréal',
 'Ville_Nashville',
 'Ville_New Orleans',
 'Ville_New York',
 'Ville_Oakland',
 'Ville_Paris',
 'Ville_Portland',
 'Ville_Québec',
 'Ville_Roma',
 'Ville_San Diego',
 'Ville_San Francisco',
 'Ville_Seattle',
 'Ville_Sydney',
 'Ville_Toronto',
 'Ville_Vancouver',
 'Ville_Venezia',
 'Ville_Washington',
 'Ville_Wien',
 'Nb_sdb',
 'Nb_chambres',
 'Nb_lits',
 'Nb_invites_inclus',
 'Nb_min_nuits',
 'dist_centr',
 'Prix_moyen',
 '24-hour check-in',
 'self check-in',
 'iron',
 'air conditioning',
 'smoke detector',
 'carbon monoxide detector',
 'cable tv',
 'dryer',
 'family/kid friendly',
 'tv']

xg_glob = pd.DataFrame(data=np.zeros(shape=(1,70)),index=[0], columns=list_col_train).astype('int')
xg_glob['Annulation_pol_'+Annulation_pol]=1
xg_glob['Type_lit_'+Type_lit]=1
xg_glob['Partage_logt_'+Partage_logt]=1
xg_glob['Type_logt_'+Type_logt]=1
xg_glob['Ville_'+ville_cour]=1
xg_glob['Nb_sdb']=Nb_sdb
xg_glob['Nb_chambres']=Nb_chambres
xg_glob['Nb_lits']=Nb_lits
xg_glob['Nb_invites_inclus']=nb_invites_inclus
xg_glob['Nb_min_nuits']=Nb_min_nuits
xg_glob['dist_centr']=dist_centr_cour
xg_glob['Prix_moyen']=prix_moyen_cour
xg_glob[Equipements]=1

ypred = XGB_global.predict(xgb.DMatrix(xg_glob.values)).astype(float)
#st.metric("Prix nuitée prédit en euros : ",str(int(round(ypred[0],2))))
st.metric(label="Prix de nuitée prédit en euros : ",value=round(ypred[0],2))
