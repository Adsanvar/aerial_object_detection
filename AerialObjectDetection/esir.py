from arcgis.gis import GIS

gis = GIS()

webmap = gis.content.get('41281c51f9de45edaf1c8ed44bb10e30')


from arcgis.mapping import WebMap
la = WebMap(webmap)

la