# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 19:36:13 2022

@author: daneb
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Visu/datasetWithAllImagesSizesEstim.csv', sep=',')
moyenne = (data.max_price + data.min_price)/2

data = pd.concat([data, moyenne], axis = 1)

######################################################################################################
df_support_cat = pd.DataFrame(columns = ['support_cat'])
df_medium_cat = pd.DataFrame(columns = ['medium_cat'])
df_cadre_cat = pd.DataFrame(columns = ['cadre_cat'])
df_lieu_cat = pd.DataFrame(columns = ['lieu_cat'])
df_qualite_cat = pd.DataFrame(columns = ['qualite_cat'])
df_reputation_cat = pd.DataFrame(columns = ['reputation_cat'])
df_datetitre_cat = pd.DataFrame(columns = ['datetitre_cat'])
df_sujet_cat = pd.DataFrame(columns = ['sujet_cat'])
df_type_cat = pd.DataFrame(columns = ['type_cat'])
df_symbolique_cat = pd.DataFrame(columns = ['symbolique'])
df_artiste_cat = pd.DataFrame(columns = ['artiste'])
df_paysage_personne = pd.DataFrame(columns = ['paysage_personne'])
df_surface = pd.DataFrame(columns = ['surface'])
##################################################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((canvas)|(toilé)|(toile))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_support_cat.loc[len(df_support_cat)] = ["toile"]
    elif re.search(r"[^a-zA-Z0-9_]((cardboard)|(papier)|(carton))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_support_cat.loc[len(df_support_cat)] = ["papier carton"]
    elif re.search(r"[^a-zA-Z0-9_]((contreplaqué)|(isorel)|(panneau))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_support_cat.loc[len(df_support_cat)] = ["panneau"]
    elif re.search(r"[^a-zA-Z0-9_]((tapisserie)|(verre)|(soie)|(cuivre))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_support_cat.loc[len(df_support_cat)] = ["autre"]
    else:
        df_support_cat.loc[len(df_support_cat)] = ["none"]
        
data = pd.concat([data, df_support_cat], axis = 1)
###########################################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((acrylique)|(huile)|(oil))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_medium_cat.loc[len(df_medium_cat)] = ["huile/acrylique"]
    elif re.search(r"[^a-zA-Z0-9_]((aquarelle)|(pastel))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_medium_cat.loc[len(df_medium_cat)] = ["aquarelle/crayons/pastelle"]
    elif re.search(r"[^a-zA-Z0-9_]((collage)|(mixte))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_medium_cat.loc[len(df_medium_cat)] = ["mixte"]
    elif re.search(r"[^a-zA-Z0-9_]((crayon)|(gravure)|(encre)|(plomb)|(fusain))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_medium_cat.loc[len(df_medium_cat)] = ["autre"]
    elif re.search(r"[^a-zA-Z0-9_](gouache)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_medium_cat.loc[len(df_medium_cat)] = ["gouache"]
    else:
        df_medium_cat.loc[len(df_medium_cat)] = ["none"]
        
data = pd.concat([data, df_medium_cat], axis = 1)
###########################################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((cadre)|(cadré)|(frame)|(ovale)|(stuc)|(stuqué))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_cadre_cat.loc[len(df_cadre_cat)] = ["cadré"]
    else:
        df_cadre_cat.loc[len(df_cadre_cat)] = ["non cadré"]

data = pd.concat([data, df_cadre_cat], axis = 1)
#############################################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((afrique)|(alger)|(algérie)|(arabe)|(biskra)|(casablanca)|(egypte)|(fès)|(fez)|(maroc)|(marrakech)|(sidi)|(tanger)|(tunis)|(tunisie))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["afrique"]
    elif re.search(r"[^a-zA-Z0-9_]((allemagne)|(allemand)|(berlin)|(hambourg)|(munich))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["allemagne"]
    elif re.search(r"[^a-zA-Z0-9_](londre)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["angleterre"]
    elif re.search(r"[^a-zA-Z0-9_](vienne)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["autriche"]
    elif re.search(r"[^a-zA-Z0-9_]((anvers)|(belge)|(belgique)|(bruges)|(bruxelles))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["belgique"]
    elif re.search(r"[^a-zA-Z0-9_]((espagne)|(espagnol)|(madrid))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["espagne"]
    elif re.search(r"[^a-zA-Z0-9_]((antibes)|(avignon)|(bordeaux)|(bretagne)|(cannes)|(chartres)|(concarneau)|(corse)|(courbevoie)|(deauville)|(dieppe)|(finistère)|(honfleur)|(lyon)|(marseille)|(martigues)|(nantes)|(neuilly)|(oise)|(orléans)|(paris)|(puy)|(rouen)|(toulon)|(toulouse)|(tréport)|(tropez)|(trouville)|(var)|(vence)|(versailles)|(yssingeaux))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["France"]
    elif re.search(r"[^a-zA-Z0-9_](grèce)(|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["grèce"]
    elif re.search(r"[^a-zA-Z0-9_](hongrois)(|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["hongrie"]
    elif re.search(r"[^a-zA-Z0-9_]((florence)|(italie)|(milan)|(naples)|(rome)|(turin)|(venise))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["italie"]
    elif re.search(r"[^a-zA-Z0-9_](israël)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["moyen orient"]
    elif re.search(r"[^a-zA-Z0-9_]((amsterdam)|(hollandais)|(hollande))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["netherland"]
    elif re.search(r"[^a-zA-Z0-9_]((asie)|(chinois)|(indochine)|(japon)|(japonais)|(orient)|(tokyo))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["orient"]
    elif re.search(r"[^a-zA-Z0-9_](polonais)(|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["pologne"]
    elif re.search(r"[^a-zA-Z0-9_]((bosphore)|(turc)|(turquie))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["proche orient"]
    elif re.search(r"[^a-zA-Z0-9_]((cyrillique)|(moscou)|(pétersbourg)|(russe)|(russie))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["russie"]
    elif re.search(r"[^a-zA-Z0-9_]((genève)|(lausanne)|(suisse)|(zurich))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["suisse"]
    elif re.search(r"[^a-zA-Z0-9_](york)(|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["usa"]
    else:
        df_lieu_cat.loc[len(df_lieu_cat)] = ["none"]

data = pd.concat([data, df_lieu_cat], axis = 1)
######################################################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((craquelure)|(écaillure)|(encrassé)|(griffe)|(griffure)|(manque)|(salissure)|(trace)|(usure))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_qualite_cat.loc[len(df_qualite_cat)] = ["abimé"]
    elif re.search(r"[^a-zA-Z0-9_]((chassis)|(châssis)|(entoilé)|(rentoilage)|(restauration)|(entoil))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_qualite_cat.loc[len(df_qualite_cat)] = ["réparé"]
    elif re.search(r"[^a-zA-Z0-9_]((accident)|(déchirure)|(enfoncement)|(humidité)|(trou))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_qualite_cat.loc[len(df_qualite_cat)] = ["très abimé"]
    else:
        df_qualite_cat.loc[len(df_qualite_cat)] = ["none"]

data = pd.concat([data, df_qualite_cat], axis = 1)
############################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((catalogue)|(musée)|(museum)|(numéro)|(raisonn))(|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_reputation_cat.loc[len(df_reputation_cat)] = ["recherché"]
    elif re.search(r"[^a-zA-Z0-9_](bibliographi)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_reputation_cat.loc[len(df_reputation_cat)] = ["reconnu"]
    elif re.search(r"[^a-zA-Z0-9_]((collection)|(exposition)|(galerie)|(gallerie)|(gallery))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_reputation_cat.loc[len(df_reputation_cat)] = ["réputé"]
    elif re.search(r"[^a-zA-Z0-9_]((académie)|(atelier)|(attestation)|(authenticit)|(cachet)|(certificat)|(inventaire)|(origine)|(particulière)|(provenance)|(tampon))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_reputation_cat.loc[len(df_reputation_cat)] = ["traçable"]
    else:
        df_reputation_cat.loc[len(df_reputation_cat)] = ["none"]

data = pd.concat([data, df_reputation_cat], axis = 1)
##############################################################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((circa)|(daté)|(dated))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_datetitre_cat.loc[len(df_datetitre_cat)] = ["daté"]
    elif re.search(r"[^a-zA-Z0-9_]((titled)|(titr)|(titré))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_datetitre_cat.loc[len(df_datetitre_cat)] = ["titre"]
    else:
        df_datetitre_cat.loc[len(df_datetitre_cat)] = ["none"]

data = pd.concat([data, df_datetitre_cat], axis = 1)
####################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((abstraction)|(abstraite)|(bonheur)|(composition)|(contemporain)|(cubisme)|(cubiste)|(été)|(hiver)|(moderne)|(vision))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_sujet_cat.loc[len(df_sujet_cat)] = ["abstrait"]
    elif re.search(r"[^a-zA-Z0-9_]((abbaye)|(allée)|(allégorie)|(allongée)|(amour)|(âne)|(anémones)|(ange)|(anglais)|(animaux)|(animé)|(antisemitism)|(arbre)|(architecture)|(arlequin)|(arme)|(assiette)|(assise)|(atlas)|(auberge)|(autoportrait)|(baie)|(baigneuse)|(bal)|(barbizon)|(barque)|(basque)|(bataille)|(bateau)|(berbère)|(berger)|(bergère)|(bergerie)|(bibliothèque)|(bijoux)|(blanche)|(blé)|(bois)|(bonaparte)|(bord)|(boucher)|(boudin)|(boulevard)|(boulogne)|(bouquet)|(bouteille)|(bretagne)|(bretonne)|(brune)|(buisson)|(buste)|(café)|(calèche)|(camp)|(canard)|(cap)|(caravane)|(casbah)|(cascade)|(cathédrale)|(cavalerie)|(cavalier)|(cerf)|(cersie)|(chaise)|(champ)|(chapeau)|(chapelle)|(chasse)|(chat)|(château)|(chaumière)|(chemin)|(chêne)|(cheval)|(chevaux)|(chien)|(christ)|(ciel)|(cirque)|(claire)|(clairière)|(classique)|(clocher)|(coiffe)|(collier)|(colline)|(coloniale)|(combat)|(commandant)|(compotier)|(comtesse)|(concert)|(coq)|(coquelicots)|(corps)|(costume)|(couchant)|(couple)|(course)|(crépuscule)|(croix)|(cruche)|(cueillette)|(cyprès)|(dahlias)|(dame)|(danse)|(déjeuner)|(désert)|(diane)|(dieu)|(diptyque)|(dreyfus)|(duchesse)|(eglise)|(église)|(elégante)|(empereur)|(enfant)|(enneig)|(espagnole)|(étang)|(etang)|(europe)|(falaise)|(fauteuil)|(femme)|(ferme)|(fête)|(feu)|(fille)|(flammande)|(fleur)|(fontaine)|(forêt)|(foulard)|(foule)|(fruit)|(fumeur)|(glace)|(glise)|(gondole)|(grève)|(guerre)|(guerrier)|(guitare)|(habit)|(hagada)|(haggada)|(hameau)|(hannoucca)|(hannuca)|(hannuka)|(hanouccah)|(harem)|(hebraica)|(hébreux)|(hebrew)|(hollandaise)|(homme)|(hussard)|(ile)|(impératice)|(impériale)|(impressionnisme)|(impressionniste)|(infanterie)|(israélite)|(jardin)|(jérusalem)|(jésus)|(jeune)|(jeux)|(jewish)|(joueur)|(journal)|(juda)|(judaisme)|(juden)|(juif)|(kasbah)|(ketouba)|(lac)|(laid)|(lapin)|(lavandières)|(légion)|(lieutenant)|(lièvre)|(lilas)|(lion)|(loing)|(loire)|(lune)|(luxembourg)|(lys)|(mains)|(maison)|(major)|(mandoline)|(manteau)|(marais)|(marchande)|(marché)|(mare)|(maréchal)|(marée)|(margeurite)|(mariage)|(marin)|(maritime)|(marne)|(mas)|(maternité)|(mâts)|(méditerrannée)|(meguila)|(menora)|(mer)|(mère)|(meuble)|(meule)|(militaire)|(miniature)|(miroir)|(moisson)|(montagne)|(montmartre)|(morte)|(moulin)|(mouton)|(musicien)|(napol)|(narcisse)|(nature)|(navire)|(neige)|(noël)|(nu)|(nuit)|(oasis)|(odalisque)|(officier)|(oiseau)|(olivier)|(ombre)|(ông)|(opéra)|(orage)|(orient)|(oued)|(panorama)|(paquebot)|(parc)|(parisienne)|(passage)|(passover)|(pâturage)|(paysage)|(paysan)|(pêche)|(péniche)|(perle)|(personnage)|(pessach)|(pessah)|(piano)|(pins)|(pivoines)|(place)|(plage)|(plaine)|(pluie)|(plume)|(poète)|(poire)|(poisson)|(pomme)|(port)|(poulailler)|(poules)|(poupée)|(prière)|(prince)|(procession)|(promenade)|(promeneur)|(promeneuse)|(provence)|(province)|(quais)|(rabbin)|(raisin)|(rayure)|(régiment)|(rempart)|(renaissance)|(république)|(rêve)|(révolution)|(rivage)|(rivière)|(robe)|(rocher)|(rocheuse)|(romantique)|(rose)|(route)|(royal)|(ruban)|(rue)|(ruine)|(ruisseau)|(sable)|(sainte)|(salon)|(sefarad)|(sein)|(sens)|(sente)|(sepharade)|(soldat)|(soleil)|(souk)|(source)|(surréaliste)|(synagogue)|(table)|(tapis)|(taverne)|(tempête)|(temple)|(tertre)|(théâtre)|(tissu)|(toilette)|(toits)|(torrent)|(tradition)|(triomphe)|(troupeau)|(trumeau)|(tuileries)|(tulipes)|(turban)|(vache)|(vague)|(vase)|(vent)|(vénus)|(verger)|(versailles)|(vieille)|(vierge)|(vieux)|(village)|(ville)|(vin)|(visage)|(voile)|(voilier)|(voyage)|(vue)|(barbizon)|(flamande)|(fontainebleau)|(hollandaise))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_sujet_cat.loc[len(df_sujet_cat)] = ["figuratif"]
    else:
        df_sujet_cat.loc[len(df_sujet_cat)] = ["none"]

data = pd.concat([data, df_sujet_cat], axis = 1)
####################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((âne)|(animaux)|(cerf)|(chat)|(cheval)|(chevaux)|(chien)|(coq)|(ferme)|(lapin)|(lièvre)|(lion)|(vache))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_type_cat.loc[len(df_type_cat)] = ["animaux"]
    elif re.search(r"[^a-zA-Z0-9_]((abbaye)|(allée)|(architecture)|(auberge)|(balcon)|(bibliothèque)|(cathédrale)|(chapelle)|(château)|(chaumière)|(eglise)|(église)|(maison)|(synagogue)|(taverne)|(temple)|(théâtre))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_type_cat.loc[len(df_type_cat)] = ["batiments"]
    elif re.search(r"[^a-zA-Z0-9_]((baie)|(baigneuse)|(cascade)|(etang)|(étang)|(fleuve)|(fontaine)|(lac)|(mare)|(marée)|(méditerranée)|(mer)|(oiseau)|(plage)|(pluie)|(rivage)|(rivière)|(ruisseau)|(troupeau)|(vagues))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_type_cat.loc[len(df_type_cat)] = ["eaux"]
    elif re.search(r"[^a-zA-Z0-9_]((arbre)|(bois)|(buisson)|(champ)|(chêne)|(feuillages)|(feuille)|(fleur)|(forêt)|(jardin))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_type_cat.loc[len(df_type_cat)] = ["plante"]
    else:
        df_type_cat.loc[len(df_type_cat)] = ["none"]

data = pd.concat([data, df_type_cat], axis = 1)
#################################################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((abbaye)|(ange)|(antisemitism)|(chapelle)|(christ)|(croix)|(dieu)|(eglise)|(église)|(jérusalem)|(jésus)|(juda)|(juif)|(noël)|(prière))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_symbolique_cat.loc[len(df_symbolique_cat)] = ["religieux"]
    elif re.search(r"[^a-zA-Z0-9_]((amour)|(bonheur)|(couple)|(mariage)|(romantique)|(rose))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_symbolique_cat.loc[len(df_symbolique_cat)] = ["sentiments"]
    else:
        df_symbolique_cat.loc[len(df_symbolique_cat)] = ["none"]

data = pd.concat([data, df_symbolique_cat], axis = 1)
########################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((apocryphe)|(attribu))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_artiste_cat.loc[len(df_artiste_cat)] = ["apocryphe"]
    elif re.search(r"[^a-zA-Z0-9_]((anonyme)|(ecole)|(école)|(entourage)|(esprit)|(illisible)|(manière))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_artiste_cat.loc[len(df_artiste_cat)] = ["inconnu"]
    elif re.search(r"[^a-zA-Z0-9_]((initiales)|(monogramm))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_artiste_cat.loc[len(df_artiste_cat)] = ["monogrammé"]
    elif re.search(r"[^a-zA-Z0-9_]((bernheim)|(camoin)|(cézanne)|(chaminade)|(contresign)|(corot)|(countersigned)|(courbet)|(delacroix)|(dupuy)|(foujita)|(friesz)|(gaugin)|(géricault)|(kisling)|(matisse)|(millet)|(monet)|(montezin)|(neillot)|(pho)|(picasso)|(pissarro)|(rasky)|(renoir)|(rousseau)|(rubens)|(saâda)|(sign)|(terzian)|(utrillo)|(valtat)|(vernet))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_artiste_cat.loc[len(df_artiste_cat)] = ["signé"]
    else:
        df_artiste_cat.loc[len(df_artiste_cat)] = ["none"]
        
data = pd.concat([data, df_artiste_cat], axis = 1)
###########################################################################
for valeur in data.complete_description:
    valeur = str(valeur).lower()
    
    if re.search(r"[^a-zA-Z0-9_]((animé)|(capitaine)|(caravane)|(cavalerie)|(cavalier)|(chasseur)|(commandant)|(danse)|(foule)|(guerre)|(guerrier)|(hussards)|(infanterie)|(lavandières)|(légion)|(marché)|(mariage)|(militaire)|(personnage)|(promeneur)|(promeneuse)|(régiment))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_paysage_personne.loc[len(df_paysage_personne)] = ["animé"]
    elif re.search(r"[^a-zA-Z0-9_]((autoportrait)|(baigneuse)|(berbère)|(berger)|(bergère)|(bonaparte)|(bretonne)|(comtesse)|(dame)|(dreyfus)|(duchesse)|(elégante)|(élégante)|(empereur)|(enfant)|(espagnole)|(femme)|(fille)|(fumeur)|(homme)|(impératrice)|(jeune)|(lieutenant)|(major)|(marchande)|(maréchal)|(marin)|(mère)|(napol)|(officier)|(parisienne)|(paysan)|(pêcheur)|(prince)|(soldat)|(villageois)|(visage))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_paysage_personne.loc[len(df_paysage_personne)] = ["humain"]
    elif re.search(r"[^a-zA-Z0-9_](bouquet)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_paysage_personne.loc[len(df_paysage_personne)] = ["nature morte"]
    elif re.search(r"[^a-zA-Z0-9_]((nu)|(nus)|(nue))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_paysage_personne.loc[len(df_paysage_personne)] = ["nu"]
    elif re.search(r"[^a-zA-Z0-9_](portrait)(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_paysage_personne.loc[len(df_paysage_personne)] = ["portrait"]
    elif re.search(r"[^a-zA-Z0-9_]((abbaye)|(allée)|(automne)|(barbizon)|(barque)|(bateau)|(bergerie)|(blé)|(bois)|(bord)|(boulevard)|(boulogne)|(bretagne)|(buisson)|(campagne)|(campement)|(cap)|(champ)|(ciel)|(clairière)|(colline)|(couchant)|(crépuscule)|(désert)|(enneig)|(étang)|(etang)|(falaise)|(ferme)|(fête)|(flamande)|(fleuve)|(fontaine)|(forêt)|(gondole)|(grève)|(hollandaise)|(ile)|(impressionniste)|(impressionnisme)|(jardin)|(lacustre)|(loire)|(marais)|(marée)|(marine)|(maritime)|(marne)|(mas)|(méditerranée)|(mer)|(meules)|(moisson)|(montagne)|(montmartre)|(montparnasse)|(moulin)|(mouton)|(navire)|(neige)|(nuit)|(oasis)|(oued)|(panorama)|(paquebot)|(parc)|(pâturage)|(paysage)|(pêche)|(péniche)|(pins)|(place)|(plage)|(plaine)|(provence)|(province)|(quais)|(remparts)|(rivage)|(rivière)|(rocher)|(rocheuse)|(route)|(rue)|(ruines)|(seine)|(sente)|(versailles)|(village)|(ville)|(voile)|(voilier)|(vue))(|e|é|és|ées|es|s)[^a-zA-Z0-9_]", valeur) is not None:
        df_paysage_personne.loc[len(df_paysage_personne)] = ["vue"]
    else:
        df_paysage_personne.loc[len(df_paysage_personne)] = ["none"]

data = pd.concat([data, df_paysage_personne], axis = 1)

data.to_csv('Visu/datasetWithAllImagesSizesEstim.csv', index=False)
########################################################################
for valeur in data.complete_description:
    if re.search("([0-9]+\s?(,|\.)?\s?[0-9]?(.)?(x|×)(.)?[0-9]+\s?(,|\.)?\s?[0-9]?)|(h\.?\s?:?\s?[0-9]+\s?(,|\.)?\s?[0-9]?\s?[a-zA-Z]*\s?-?x?;?\.?\s?l\.?\s?:?\s?[0-9]+\s?(,|\.)?\s?[0-9]?\s?[a-zA-Z]+)", valeur) is not None:
        taille = re.search("([0-9]+\s?(,|\.)?\s?[0-9]?(.)?(x|×)(.)?[0-9]+\s?(,|\.)?\s?[0-9]?)|(h\.?\s?:?\s?[0-9]+\s?(,|\.)?\s?[0-9]?\s?[a-zA-Z]*\s?-?x?;?\.?\s?l\.?\s?:?\s?[0-9]+\s?(,|\.)?\s?[0-9]?\s?[a-zA-Z]+)", valeur).group()
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
            df_surface.loc[len(df_surface)] = [surface]
    else:
        df_surface.loc[len(df_surface)] = ['']
        

data = pd.concat([data, df_surface], axis = 1)

print(data.complete_description[0])

data.to_csv('Visu/datasetWithAllImagesSizesEstim.csv', index=False)

