# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:13:45 2022

@author: daneb
"""

from xml.etree.ElementInclude import include
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
import matplotlib.pyplot as plt
import glob
from PIL import Image
import os
import shutil
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm

embed_image = ResNet50(weights='imagenet',
                        include_top=False,
                        input_shape=(224,224,3),
                        pooling='max')

embed_text = SentenceTransformer('all-MiniLM-L6-v2')



#Modèle description
def description_predict(text):
#text=["Théodore GÉRICAULT, (1791-1824) Tête de bouledogue. circa 1818-1820 Huile sur toile, porte au verso sur le châssis une annotation « C.M. Mathieu 1876 » à l’encre. 24, 2 x 32, 2 cm (rentoilage des années 1970-1980). Sera inclus dans le Catalogue raisonné des tableaux de Théodore Géricault, actuellement en préparation par Monsieur Bruno Chenique. Provenance : - Collection de feu M. Mathieu, E Girard commissairepriseur, Féral , peintre-expert, Paris, Hotel Drouot, salle n°7, lundi 11 décembre 1876, n° 5. - Paris, collection Edmond Courty - Paris, collection particulière Une expertise de Monsieur Bruno Chenique en date du 30 novembre 2020 sera remise à l’acquéreur. Villanfray France Paris"]
  list=[]
  list.append(text)
  text_embed = embed_text.encode(list)
  model_text = keras.models.load_model('model/tfidf')
  prix_log = model_text.predict(text_embed)
  return np.exp(prix_log)

def extract_features(img_path, model):
	input_shape = (224, 224, 3)
	img = image.load_img(img_path,
                     	target_size=(input_shape[0], input_shape[1]))
	img_array = image.img_to_array(img)
	expanded_img_array = np.expand_dims(img_array, axis=0)
	preprocessed_img = preprocess_input(expanded_img_array)
	features = model.predict(preprocessed_img)
	flattened_features = features.flatten()
	normalized_features = flattened_features / norm(flattened_features)
	return normalized_features


#Test image
def image_predict(image):
  img_embed = extract_features(image, embed_image)
  model_image = keras.models.load_model('model/imgmodel_resnet50imagenet_adam_mse_2_32_500')
  prix_log = model_image.predict(np.array(img_embed))
  return np.exp(prix_log)

#Test text and image
def desc_image_predict(text, image):
  text_embed = model_text.encode(text)
  img_embed = model_image.encode([Image.open(img) for img in image], batch_size=32, show_progress_bar=True)
  total_embed = np.concatenate((text_embed, img_embed), axis=1)
  #prix_log = classifier_log.predict(total_embed)
  #return np.exp(prix_log)

def desc_significant_predict(desc):
    desc = str(desc).lower()
    surface = 0
    symbolique = "none"
    typ = "none"
    sujet = "none"
    datetitre = "none"
    reputation = 0
    qualite = "none"
    lieu = 0
    cadre = "non cadré"
    medium = 0
    support = 0
    artiste = "none"
    paysage_personne = "none"
    
    #Surface
    taille = re.search("([0-9]+\s?(,|\.)?\s?[0-9]?(.)?(x|×)(.)?[0-9]+\s?(,|\.)?\s?[0-9]?)|(h\.?\s?:?\s?[0-9]+\s?(,|\.)?\s?[0-9]?\s?[a-zA-Z]*\s?-?x?;?\.?\s?l\.?\s?:?\s?[0-9]+\s?(,|\.)?\s?[0-9]?\s?[a-zA-Z]+)", desc).group()
    m = re.findall(r"-?\d+\s?\.?\s?\d*", taille.replace(" ", "").replace(",", "."))
    if len(m) == 2:
        if float(m[0]) > float(m[1]):
            longueur = float(m[0])
            largeur = float(m[1])
        else:
            longueur = float(m[1])
            largeur = float(m[0])
        for val in m:
            surf = str(longueur * largeur).replace(".", "")
            surf = surf[0:2] + "." + surf[2:]
            surface = float(surf)
    
    #Symbolique
    if re.search(r"[^a-zA-Z0-9_]((abbaye)|(ange)|(antisemitism)|(chapelle)|(christ)|(croix)|(dieu)|(eglise)|(église)|(jérusalem)|(jésus)|(juda)|(juif)|(noël)|(prière))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        symbolique = "religieux"
    elif re.search(r"[^a-zA-Z0-9_]((amour)|(bonheur)|(couple)|(mariage)|(romantique)|(rose))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        symbolique = "sentiments"
    
    #Type
    if re.search(r"[^a-zA-Z0-9_]((âne)|(animaux)|(cerf)|(chat)|(cheval)|(chevaux)|(chien)|(coq)|(ferme)|(lapin)|(lièvre)|(lion)|(vache))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        typ = "animaux"
    elif re.search(r"[^a-zA-Z0-9_]((abbaye)|(allée)|(architecture)|(auberge)|(balcon)|(bibliothèque)|(cathédrale)|(chapelle)|(château)|(chaumière)|(eglise)|(église)|(maison)|(synagogue)|(taverne)|(temple)|(théâtre))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        typ = "batiments"
    elif re.search(r"[^a-zA-Z0-9_]((baie)|(baigneuse)|(cascade)|(etang)|(étang)|(fleuve)|(fontaine)|(lac)|(mare)|(marée)|(méditerranée)|(mer)|(oiseau)|(plage)|(pluie)|(rivage)|(rivière)|(ruisseau)|(troupeau)|(vagues))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        typ = "eaux"
    elif re.search(r"[^a-zA-Z0-9_]((arbre)|(bois)|(buisson)|(champ)|(chêne)|(feuillages)|(feuille)|(fleur)|(forêt)|(jardin))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        typ = "plante"
    
    #Sujet
    if re.search(r"[^a-zA-Z0-9_]((abstraction)|(abstraite)|(bonheur)|(composition)|(contemporain)|(cubisme)|(cubiste)|(été)|(hiver)|(moderne)|(vision))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        sujet = "abstrait"
    elif re.search(r"[^a-zA-Z0-9_]((abbaye)|(allée)|(allégorie)|(allongée)|(amour)|(âne)|(anémones)|(ange)|(anglais)|(animaux)|(animé)|(antisemitism)|(arbre)|(architecture)|(arlequin)|(arme)|(assiette)|(assise)|(atlas)|(auberge)|(autoportrait)|(baie)|(baigneuse)|(bal)|(barbizon)|(barque)|(basque)|(bataille)|(bateau)|(berbère)|(berger)|(bergère)|(bergerie)|(bibliothèque)|(bijoux)|(blanche)|(blé)|(bois)|(bonaparte)|(bord)|(boucher)|(boudin)|(boulevard)|(boulogne)|(bouquet)|(bouteille)|(bretagne)|(bretonne)|(brune)|(buisson)|(buste)|(café)|(calèche)|(camp)|(canard)|(cap)|(caravane)|(casbah)|(cascade)|(cathédrale)|(cavalerie)|(cavalier)|(cerf)|(cersie)|(chaise)|(champ)|(chapeau)|(chapelle)|(chasse)|(chat)|(château)|(chaumière)|(chemin)|(chêne)|(cheval)|(chevaux)|(chien)|(christ)|(ciel)|(cirque)|(claire)|(clairière)|(classique)|(clocher)|(coiffe)|(collier)|(colline)|(coloniale)|(combat)|(commandant)|(compotier)|(comtesse)|(concert)|(coq)|(coquelicots)|(corps)|(costume)|(couchant)|(couple)|(course)|(crépuscule)|(croix)|(cruche)|(cueillette)|(cyprès)|(dahlias)|(dame)|(danse)|(déjeuner)|(désert)|(diane)|(dieu)|(diptyque)|(dreyfus)|(duchesse)|(eglise)|(église)|(elégante)|(empereur)|(enfant)|(enneig)|(espagnole)|(étang)|(etang)|(europe)|(falaise)|(fauteuil)|(femme)|(ferme)|(fête)|(feu)|(fille)|(flammande)|(fleur)|(fontaine)|(forêt)|(foulard)|(foule)|(fruit)|(fumeur)|(glace)|(glise)|(gondole)|(grève)|(guerre)|(guerrier)|(guitare)|(habit)|(hagada)|(haggada)|(hameau)|(hannoucca)|(hannuca)|(hannuka)|(hanouccah)|(harem)|(hebraica)|(hébreux)|(hebrew)|(hollandaise)|(homme)|(hussard)|(ile)|(impératice)|(impériale)|(impressionnisme)|(impressionniste)|(infanterie)|(israélite)|(jardin)|(jérusalem)|(jésus)|(jeune)|(jeux)|(jewish)|(joueur)|(journal)|(juda)|(judaisme)|(juden)|(juif)|(kasbah)|(ketouba)|(lac)|(laid)|(lapin)|(lavandières)|(légion)|(lieutenant)|(lièvre)|(lilas)|(lion)|(loing)|(loire)|(lune)|(luxembourg)|(lys)|(mains)|(maison)|(major)|(mandoline)|(manteau)|(marais)|(marchande)|(marché)|(mare)|(maréchal)|(marée)|(margeurite)|(mariage)|(marin)|(maritime)|(marne)|(mas)|(maternité)|(mâts)|(méditerrannée)|(meguila)|(menora)|(mer)|(mère)|(meuble)|(meule)|(militaire)|(miniature)|(miroir)|(moisson)|(montagne)|(montmartre)|(morte)|(moulin)|(mouton)|(musicien)|(napol)|(narcisse)|(nature)|(navire)|(neige)|(noël)|(nu)|(nuit)|(oasis)|(odalisque)|(officier)|(oiseau)|(olivier)|(ombre)|(ông)|(opéra)|(orage)|(orient)|(oued)|(panorama)|(paquebot)|(parc)|(parisienne)|(passage)|(passover)|(pâturage)|(paysage)|(paysan)|(pêche)|(péniche)|(perle)|(personnage)|(pessach)|(pessah)|(piano)|(pins)|(pivoines)|(place)|(plage)|(plaine)|(pluie)|(plume)|(poète)|(poire)|(poisson)|(pomme)|(port)|(poulailler)|(poules)|(poupée)|(prière)|(prince)|(procession)|(promenade)|(promeneur)|(promeneuse)|(provence)|(province)|(quais)|(rabbin)|(raisin)|(rayure)|(régiment)|(rempart)|(renaissance)|(république)|(rêve)|(révolution)|(rivage)|(rivière)|(robe)|(rocher)|(rocheuse)|(romantique)|(rose)|(route)|(royal)|(ruban)|(rue)|(ruine)|(ruisseau)|(sable)|(sainte)|(salon)|(sefarad)|(sein)|(sens)|(sente)|(sepharade)|(soldat)|(soleil)|(souk)|(source)|(surréaliste)|(synagogue)|(table)|(tapis)|(taverne)|(tempête)|(temple)|(tertre)|(théâtre)|(tissu)|(toilette)|(toits)|(torrent)|(tradition)|(triomphe)|(troupeau)|(trumeau)|(tuileries)|(tulipes)|(turban)|(vache)|(vague)|(vase)|(vent)|(vénus)|(verger)|(versailles)|(vieille)|(vierge)|(vieux)|(village)|(ville)|(vin)|(visage)|(voile)|(voilier)|(voyage)|(vue)|(barbizon)|(flamande)|(fontainebleau)|(hollandaise))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        sujet = "figuratif"
    
    #Date ou titre
    if re.search(r"[^a-zA-Z0-9_]((circa)|(daté)|(dated))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        datetitre = "daté"
    elif re.search(r"[^a-zA-Z0-9_]((titled)|(titr)|(titré))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        datetitre = "titre"
        
    #Reputation
    if re.search(r"[^a-zA-Z0-9_]((catalogue)|(musée)|(museum)|(numéro)|(raisonn))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        reputation = 4
    elif re.search(r"[^a-zA-Z0-9_](bibliographie)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        reputation = 2
    elif re.search(r"[^a-zA-Z0-9_]((collection)|(exposition)|(galerie)|(gallerie)|(gallery))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        reputation = 3
    elif re.search(r"[^a-zA-Z0-9_]((académie)|(atelier)|(attestation)|(authenticit)|(cachet)|(certificat)|(inventaire)|(origine)|(particulière)|(provenance)|(tampon))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        reputation = 1
    
    #Qualité
    if re.search(r"[^a-zA-Z0-9_]((craquelure)|(écaillure)|(encrassé)|(griffe)|(griffure)|(manque)|(salissure)|(trace)|(usure))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        qualite = "abimé"
    elif re.search(r"[^a-zA-Z0-9_]((chassis)|(châssis)|(entoilé)|(rentoilage)|(restauration)|(entoil))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        qualite = "réparé"
    elif re.search(r"[^a-zA-Z0-9_]((accident)|(déchirure)|(enfoncement)|(humidité)|(trou))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        qualite = "très abimé"
        
    #Lieu
    if re.search(r"[^a-zA-Z0-9_]((afrique)|(alger)|(algérie)|(arabe)|(biskra)|(casablanca)|(egypte)|(fès)|(fez)|(maroc)|(marrakech)|(sidi)|(tanger)|(tunis)|(tunisie))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 7
    elif re.search(r"[^a-zA-Z0-9_]((allemagne)|(allemand)|(berlin)|(hambourg)|(munich))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 11
    elif re.search(r"[^a-zA-Z0-9_](londre)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 13
    elif re.search(r"[^a-zA-Z0-9_](vienne)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 10
    elif re.search(r"[^a-zA-Z0-9_]((anvers)|(belge)|(belgique)|(bruges)|(bruxelles))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 15
    elif re.search(r"[^a-zA-Z0-9_]((espagne)|(espagnol)|(madrid))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 12
    elif re.search(r"[^a-zA-Z0-9_]((antibes)|(avignon)|(bordeaux)|(bretagne)|(cannes)|(chartres)|(concarneau)|(corse)|(courbevoie)|(deauville)|(dieppe)|(finistère)|(honfleur)|(lyon)|(marseille)|(martigues)|(nantes)|(neuilly)|(oise)|(orléans)|(paris)|(puy)|(rouen)|(toulon)|(toulouse)|(tréport)|(tropez)|(trouville)|(var)|(vence)|(versailles)|(yssingeaux))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 18
    elif re.search(r"[^a-zA-Z0-9_](grèce)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 6
    elif re.search(r"[^a-zA-Z0-9_](hongrois)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 8
    elif re.search(r"[^a-zA-Z0-9_]((florence)|(italie)|(milan)|(naples)|(rome)|(turin)|(venise))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 9
    elif re.search(r"[^a-zA-Z0-9_](israël)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 4
    elif re.search(r"[^a-zA-Z0-9_]((amsterdam)|(hollandais)|(hollande))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 14
    elif re.search(r"[^a-zA-Z0-9_]((asie)|(chinois)|(indochine)|(japon)|(japonais)|(orient)|(tokyo))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 1
    elif re.search(r"[^a-zA-Z0-9_](polonais)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 17
    elif re.search(r"[^a-zA-Z0-9_]((bosphore)|(turc)|(turquie))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 5
    elif re.search(r"[^a-zA-Z0-9_]((cyrillique)|(moscou)|(pétersbourg)|(russe)|(russie))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 3
    elif re.search(r"[^a-zA-Z0-9_]((genève)|(lausanne)|(suisse)|(zurich))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 16
    elif re.search(r"[^a-zA-Z0-9_](york)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        lieu = 2
        
    #Cadre
    if re.search(r"[^a-zA-Z0-9_]((cadre)|(cadré)|(frame)|(ovale)|(stuc)|(stuqué))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        cadre = "cadré"
    
    #Medium
    if re.search(r"[^a-zA-Z0-9_]((collage)|(mixte))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        medium = 4
    elif re.search(r"[^a-zA-Z0-9_](gouache)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        medium = 3
    elif re.search(r"[^a-zA-Z0-9_]((oil)|(acrylique)|(huile))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        medium = 5
    elif re.search(r"[^a-zA-Z0-9_]((pastel)|(aquarelle))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        medium = 2
    elif re.search(r"[^a-zA-Z0-9_]((plomb)|(encre)|(crayon)|(fusain)|(gravure))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        medium = 1
    
    #Support
    if re.search(r"[^a-zA-Z0-9_]((canvas)|(toile)|(toilé))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        support = 3
    elif re.search(r"[^a-zA-Z0-9_]((cardboard)|(carton)|(papier))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        support = 4
    elif re.search(r"[^a-zA-Z0-9_]((contreplaqué)|(isorel)|(panneau))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        support = 2
    elif re.search(r"[^a-zA-Z0-9_]((soie)|(cuivre)|(tapisserie)|(verre))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        support = 1
    
    #Artiste  
    if re.search(r"[^a-zA-Z0-9_]((apocryphe)|(attribu))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        artiste = "apocryphe"
    elif re.search(r"[^a-zA-Z0-9_]((anonyme)|(ecole)|(école)|(entourage)|(esprit)|(illisible)|(manière))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        artiste = "inconnu"
    elif re.search(r"[^a-zA-Z0-9_]((initiales)|(monogramm))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        artiste = "monogrammé"
    elif re.search(r"[^a-zA-Z0-9_]((bernheim)|(camoin)|(cézanne)|(chaminade)|(contresign)|(corot)|(countersigned)|(courbet)|(delacroix)|(dupuy)|(foujita)|(friesz)|(gaugin)|(géricault)|(kisling)|(matisse)|(millet)|(monet)|(montezin)|(neillot)|(pho)|(picasso)|(pissarro)|(rasky)|(renoir)|(rousseau)|(rubens)|(saâda)|(sign)|(terzian)|(utrillo)|(valtat)|(vernet))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        artiste = "signé"
    
            
    
    #Paysage
    if re.search(r"[^a-zA-Z0-9_]((animé)|(capitaine)|(caravane)|(cavalerie)|(cavalier)|(chasseur)|(commandant)|(danse)|(foule)|(guerre)|(guerrier)|(hussards)|(infanterie)|(lavandières)|(légion)|(marché)|(mariage)|(militaire)|(personnage)|(promeneur)|(promeneuse)|(régiment))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        paysage_personne = "animé"
    elif re.search(r"[^a-zA-Z0-9_]((autoportrait)|(baigneuse)|(berbère)|(berger)|(bergère)|(bonaparte)|(bretonne)|(comtesse)|(dame)|(dreyfus)|(duchesse)|(elégante)|(élégante)|(empereur)|(enfant)|(espagnole)|(femme)|(fille)|(fumeur)|(homme)|(impératrice)|(jeune)|(lieutenant)|(major)|(marchande)|(maréchal)|(marin)|(mère)|(napol)|(officier)|(parisienne)|(paysan)|(pêcheur)|(prince)|(soldat)|(villageois)|(visage))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        paysage_personne = "humain"
    elif re.search(r"[^a-zA-Z0-9_](bouquet)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        paysage_personne = "nature morte"
    elif re.search(r"[^a-zA-Z0-9_]((nu)|(nus)|(nue))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        paysage_personne = "nu"
    elif re.search(r"[^a-zA-Z0-9_](portrait)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        paysage_personne = "portrait"
    elif re.search(r"[^a-zA-Z0-9_]((abbaye)|(allée)|(automne)|(barbizon)|(barque)|(bateau)|(bergerie)|(blé)|(bois)|(bord)|(boulevard)|(boulogne)|(bretagne)|(buisson)|(campagne)|(campement)|(cap)|(champ)|(ciel)|(clairière)|(colline)|(couchant)|(crépuscule)|(désert)|(enneig)|(étang)|(etang)|(falaise)|(ferme)|(fête)|(flamande)|(fleuve)|(fontaine)|(forêt)|(gondole)|(grève)|(hollandaise)|(ile)|(impressionniste)|(impressionnisme)|(jardin)|(lacustre)|(loire)|(marais)|(marée)|(marine)|(maritime)|(marne)|(mas)|(méditerranée)|(mer)|(meules)|(moisson)|(montagne)|(montmartre)|(montparnasse)|(moulin)|(mouton)|(navire)|(neige)|(nuit)|(oasis)|(oued)|(panorama)|(paquebot)|(parc)|(pâturage)|(paysage)|(pêche)|(péniche)|(pins)|(place)|(plage)|(plaine)|(provence)|(province)|(quais)|(remparts)|(rivage)|(rivière)|(rocher)|(rocheuse)|(route)|(rue)|(ruines)|(seine)|(sente)|(versailles)|(village)|(ville)|(voile)|(voilier)|(vue))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", desc) is not None:
        paysage_personne = "vue"

    return surface, symbolique, typ, sujet, datetitre, reputation, qualite, lieu, cadre, medium, support, artiste, paysage_personne
