"""Rebuild the NCITIES spatial analysis from point GDBs and contextual layers.

Run with the ArcGIS Pro Python environment (its GDAL bindings work without an
ArcGIS licence):

  "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python.exe" ncities_analysis.py

The input GDBs already contain corrected population positions and fused,
classified restaurants. Original geocoding, web scraping, cross-source fusion,
photographs and basemaps cannot be recovered from them. The script reports
reproduction differences rather than silently changing the published results.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse, stats
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, curve_fit
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree, distance

PROVINCES_ZH = ["广东", "湖南", "湖北", "广西", "江西", "河南", "四川", "福建", "安徽", "贵州", "重庆", "黑龙江", "浙江", "山东", "海南", "陕西", "江苏", "云南", "辽宁", "河北", "吉林", "甘肃", "山西", "新疆", "内蒙古", "青海", "宁夏", "北京", "上海", "天津", "西藏"]
PROVINCES = ["Guangdong", "Hunan", "Hubei", "Guangxi", "Jiangxi", "Henan", "Sichuan", "Fujian", "Anhui", "Guizhou", "Chongqing", "Heilongjiang", "Zhejiang", "Shandong", "Hainan", "Shaanxi", "Jiangsu", "Yunnan", "Liaoning", "Hebei", "Jilin", "Gansu", "Shanxi", "Xinjiang", "Inner Mongolia", "Qinghai", "Ningxia", "Beijing", "Shanghai", "Tianjin", "Tibet"]
CUISINES_ZH = ["广东", "湖南", "湖北", "广西", "江西", "河南", "四川", "福建", "重庆", "陕西", "甘肃", "新疆", "北京", "东北", "西北", "云贵"]
CUISINES = ["Guangdong", "Hunan", "Hubei", "Guangxi", "Jiangxi", "Henan", "Sichuan", "Fujian", "Chongqing", "Shaanxi", "Gansu", "Xinjiang", "Beijing", "Northeast", "Northwest", "Yunnan–Guizhou"]
MEMBERS = {c: [c] for c in CUISINES[:13]}
MEMBERS.update({"Northeast": ["Liaoning", "Jilin", "Heilongjiang"], "Northwest": ["Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang"], "Yunnan–Guizhou": ["Yunnan", "Guizhou"]})
MATCH = np.array([[int(p in MEMBERS[c]) for c in CUISINES] for p in PROVINCES], dtype=np.int8)
ZH_P = {x: i for i, x in enumerate(PROVINCES_ZH)}
ZH_C = {x: i for i, x in enumerate(CUISINES_ZH)}
SCALES = tuple(range(100, 1001, 100))


def args_parser():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--population-gdb", type=Path, default=Path("population_corrected_fullsample.gdb/population_corrected_fullsample.gdb"))
    a.add_argument("--restaurant-gdb", type=Path, default=Path("restaurants_merged.gdb/restaurants_merged.gdb"))
    a.add_argument("--restaurant-vector", type=Path, default=Path("variables/regional_cuisine_restaurants.gpkg"), help="Fallback when the merged restaurant GDB is unavailable")
    a.add_argument("--out", type=Path, default=Path("analysis_output"))
    a.add_argument("--permutations", type=int, default=9999)
    a.add_argument("--scale-permutations", type=int, default=999)
    a.add_argument("--matrix-permutations", type=int, default=9999)
    a.add_argument("--seed", type=int, default=20261120)
    a.add_argument("--limit-pop", type=int, default=0, help="Only for a smoke test; never for reported results")
    a.add_argument("--x-origin", type=float, default=0)
    a.add_argument("--y-origin", type=float, default=0)
    a.add_argument("--prepare-variables", action="store_true", help="Clip and export contextual layers into variables/")
    a.add_argument("--prepare-only", action="store_true", help="Prepare variables without rerunning the population analysis")
    a.add_argument("--variables-dir", type=Path, default=Path("variables"))
    a.add_argument("--cbd-longitude", type=float, default=113.319)
    a.add_argument("--cbd-latitude", type=float, default=23.124)
    a.add_argument("--skip-context", action="store_true", help="Run only analyses requiring the two point GDBs")
    return a.parse_args()


def prepare_variables(a):
    from osgeo import ogr, osr, gdal
    boundary_path=Path(r"G:\Vincent\论文实验-昼夜热暴露\研究区界.shp")
    specs=[
        ("regional_cuisine_restaurants",a.restaurant_gdb,"restaurant_points",None,["restaurant_id","source","source_id","name","normalized_name","category","cuisine","match_provinces","sources_present","cuisine_code","method","easting","northing"]),
        ("metro_stations",Path(r"G:\DATA\广州地铁站\广州地铁站.shp"),None,None,["站点","线路"]),
        ("roads",Path(r"G:\DATA\粤港澳大湾区数据集\大湾区路网\大湾区路网.shp"),None,None,["fclass"]),
        ("non_restaurant_pois",Path(r"G:\DATA\广州POIs\GZ_POIs_wgs84.shp"),None,"not_restaurant",["一级分类","二级分类"]),
        ("shopping_pois",Path(r"G:\DATA\广州POIs\GZ_POIs_wgs84.shp"),None,"shopping",["一级分类","二级分类"]),
        ("housing_transactions_2019",Path(r"F:\论文实验-人口模拟\r1\二手房价.shp"),None,"housing_2019",["成交时","每平米","房屋用"]),
        ("building_footprints",Path(r"G:\S-DATA\广东省建筑面\广东省建筑面.gdb"),"广州",None,["实际层","占地面"]),
    ]
    a.variables_dir.mkdir(parents=True,exist_ok=True)
    pop_ds,pop_layer=gdb_layer(a.population_gdb,"population_points")
    target=pop_layer.GetSpatialRef().Clone()
    boundary_ds=ogr.Open(str(boundary_path));boundary_layer=boundary_ds.GetLayer(0)
    boundary_src=boundary_layer.GetSpatialRef()
    boundary_feature=boundary_layer.GetNextFeature()
    boundary_wgs=boundary_feature.GetGeometryRef().Clone()
    boundary_proj=boundary_wgs.Clone()
    boundary_proj.Transform(osr.CoordinateTransformation(boundary_src,target))
    def new_layer(path,name,geom_type,fields):
        if path.exists():path.unlink()
        data=ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
        layer=data.CreateLayer(name,target,geom_type)
        for outname,ftype in fields:layer.CreateField(ogr.FieldDefn(outname,ftype))
        return data,layer
    bd_data,bd_layer=new_layer(a.variables_dir/"study_area.gpkg","study_area",ogr.wkbPolygon,[])
    bf=ogr.Feature(bd_layer.GetLayerDefn());bf.SetGeometry(boundary_proj);bd_layer.CreateFeature(bf)
    bd_data=None
    cbd_data,cbd_layer=new_layer(a.variables_dir/"cbd_reference_point.gpkg","cbd_reference_point",ogr.wkbPoint,[("name",ogr.OFTString)])
    cbd=ogr.Geometry(ogr.wkbPoint);cbd.AddPoint_2D(a.cbd_longitude,a.cbd_latitude)
    cbd.Transform(osr.CoordinateTransformation(boundary_src,target))
    cbd_feature=ogr.Feature(cbd_layer.GetLayerDefn());cbd_feature.SetGeometry(cbd)
    cbd_feature.SetField("name","Zhujiang New Town CBD");cbd_layer.CreateFeature(cbd_feature)
    cbd_data=None
    log={"study_area":"one clipped study polygon","cbd_lon_lat":[a.cbd_longitude,a.cbd_latitude],"layers":{}}
    lon_min,lon_max,lat_min,lat_max=boundary_wgs.GetEnvelope()
    proj_minx,proj_maxx,proj_miny,proj_maxy=boundary_proj.GetEnvelope()
    for name,src,layer_name,selector,fields in specs:
        if not src.exists():
            log["layers"][name]={"error":"missing input","path":str(src)};continue
        source=ogr.Open(str(src),0)
        layer=source.GetLayerByName(layer_name) if layer_name else source.GetLayer(0)
        if layer is None:
            log["layers"][name]={"error":"missing layer","path":str(src)};continue
        in_srs=layer.GetSpatialRef()
        if in_srs is None:raise ValueError("Missing CRS: "+str(src))
        projected=bool(in_srs.IsSame(target))
        if projected:layer.SetSpatialFilterRect(proj_minx,proj_miny,proj_maxx,proj_maxy)
        else:layer.SetSpatialFilterRect(lon_min,lat_min,lon_max,lat_max)
        tr=None if projected else osr.CoordinateTransformation(in_srs,target)
        output_names=["field_"+str(i+1) for i in range(len(fields))]
        if name=="regional_cuisine_restaurants":output_names=["restaurant_id","source","source_id","name","normalized_name","source_category","cuisine","match_provinces","sources_present","cuisine_code","classification_method","source_easting","source_northing"]
        if name=="metro_stations":output_names=["station_name","line_name"]
        if name=="roads":output_names=["road_class"]
        if name=="building_footprints":output_names=["floors","source_footprint_area"]
        if name.endswith("pois"):output_names=["poi_category","poi_subcategory"]
        if name=="housing_transactions_2019":output_names=["transaction_date","price_cny_m2","housing_use"]
        outfields=[(n,ogr.OFTInteger if n in ("cuisine_code","classification_method") else ogr.OFTReal if n in ("price_cny_m2","floors","source_footprint_area","source_easting","source_northing") else ogr.OFTString) for n in output_names]
        geom_type=ogr.wkbPoint if name not in ("roads","building_footprints") else ogr.wkbUnknown
        dest,dlayer=new_layer(a.variables_dir/(name+".gpkg"),name,geom_type,outfields)
        count=0
        dest.StartTransaction()
        for f in layer:
            category=f.GetField("一级分类") if selector in ("not_restaurant","shopping") else None
            if selector=="not_restaurant" and category=="餐饮服务":continue
            if selector=="shopping" and category!="购物服务":continue
            if selector=="housing_2019":
                date=str(f.GetField("成交时") or "")
                price=f.GetField("每平米")
                use=f.GetField("房屋用")
                if not date.startswith("2019") or price is None or price<=0 or use not in ("普通住宅","别墅"):
                    continue
            geom=f.GetGeometryRef()
            if geom is None:continue
            geom=geom.Clone();geom.FlattenTo2D()
            if tr:geom.Transform(tr)
            if not geom.Intersects(boundary_proj):continue
            if name in ("roads","building_footprints") and not boundary_proj.Contains(geom):
                if not geom.IsValid():geom=geom.Buffer(0)
                geom=geom.Intersection(boundary_proj)
                if geom is None or geom.IsEmpty():continue
                # Geometry collections can arise at the boundary; keep only the expected dimension.
                if name=="roads" and geom.GetDimension()!=1:continue
                if name=="building_footprints" and geom.GetDimension()!=2:continue
            feat=ogr.Feature(dlayer.GetLayerDefn());feat.SetGeometry(geom)
            for original,renamed in zip(fields,output_names):
                value=f.GetField(original)
                if value is not None:feat.SetField(renamed,value)
            if dlayer.CreateFeature(feat)==0:count+=1
            if count and count%25000==0:
                dest.CommitTransaction();dest.StartTransaction()
        dest.CommitTransaction();dest=None;source=None
        log["layers"][name]={"path":str((a.variables_dir/(name+".gpkg")).resolve()),"features":count,"crs":target.GetAttrValue("PROJCS")}
        print("Prepared",name,count,flush=True)
    raster_source=Path(r"D:\variable\ntl.tif")
    if raster_source.exists():
        raster_out=a.variables_dir/"nighttime_light_100m.tif"
        if raster_out.exists():raster_out.unlink()
        warped=gdal.Warp(str(raster_out),str(raster_source),format="GTiff",dstSRS=target.ExportToWkt(),cutlineDSName=str(boundary_path),cropToCutline=True,xRes=100,yRes=100,resampleAlg="average",dstNodata=-9999,creationOptions=["COMPRESS=LZW","TILED=YES"])
        if warped is None:raise RuntimeError("Nighttime-light warp failed")
        log["layers"]["nighttime_light"]={"path":str(raster_out.resolve()),"width":warped.RasterXSize,"height":warped.RasterYSize}
        warped=None
        log["layers"]["nighttime_light"]["outside_centres_masked"]=mask_raster_centres_to_boundary(raster_out,boundary_proj)
    log["layers"]["cbd_reference_point"]={"path":str((a.variables_dir/"cbd_reference_point.gpkg").resolve()),"longitude":a.cbd_longitude,"latitude":a.cbd_latitude}
    surface=housing_surface(a,boundary_proj)
    if surface:log["layers"]["housing_log_price_200m"]={"path":str(surface.resolve()),"method":"reconstructed local ordinary kriging"}
    export_restaurant_auxiliary_tables(a)
    (a.variables_dir/"source_manifest.json").write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding="utf-8")
    return log


def gdb_layer(path, name):
    try:
        from osgeo import ogr
    except ImportError as exc:
        raise SystemExit("GDAL/OGR is required; run with ArcGIS Pro's Python environment.") from exc
    ds = ogr.Open(str(path.resolve()), 0)
    if ds is None:
        raise FileNotFoundError(path)
    layer = ds.GetLayerByName(name)
    if layer is None:
        raise ValueError(f"Layer {name!r} missing in {path}")
    if layer.GetSpatialRef() is None or abs(layer.GetSpatialRef().GetLinearUnits() - 1) > 0.01:
        raise ValueError(f"{name}: expected projected metre coordinates")
    return ds, layer


def mask_raster_centres_to_boundary(path,boundary):
    """Set raster cells whose centres fall outside the exact study polygon to NoData."""
    from osgeo import ogr,gdal
    ds=gdal.Open(str(path),gdal.GA_Update);band=ds.GetRasterBand(1)
    data=band.ReadAsArray();nodata=band.GetNoDataValue();gt=ds.GetGeoTransform()
    point=ogr.Geometry(ogr.wkbPoint);changed=0
    for row in range(data.shape[0]):
        y=gt[3]+(row+.5)*gt[5]
        for col in range(data.shape[1]):
            value=data[row,col]
            if not np.isfinite(value) or (nodata is not None and value==nodata):continue
            x=gt[0]+(col+.5)*gt[1];point.SetPoint_2D(0,float(x),float(y))
            if not boundary.Intersects(point):data[row,col]=nodata;changed+=1
    band.WriteArray(data);band.FlushCache();ds=None
    return changed


def refresh_restaurant_vector(a):
    """Refresh the full restaurant attributes in the clipped GeoPackage."""
    from osgeo import ogr
    source,layer=gdb_layer(a.restaurant_gdb,"restaurant_points")
    study=ogr.Open(str(a.variables_dir/"study_area.gpkg"));study_layer=study.GetLayer(0)
    boundary_feature=study_layer.GetNextFeature();boundary=boundary_feature.GetGeometryRef().Clone()
    names=[("restaurant_id",ogr.OFTString),("source",ogr.OFTString),("source_id",ogr.OFTString),("name",ogr.OFTString),("normalized_name",ogr.OFTString),("category",ogr.OFTString),("cuisine",ogr.OFTString),("match_provinces",ogr.OFTString),("sources_present",ogr.OFTString),("cuisine_code",ogr.OFTInteger),("method",ogr.OFTInteger),("easting",ogr.OFTReal),("northing",ogr.OFTReal)]
    dest_path=a.variables_dir/"regional_cuisine_restaurants.gpkg"
    if dest_path.exists():dest_path.unlink()
    dest=ogr.GetDriverByName("GPKG").CreateDataSource(str(dest_path))
    out=dest.CreateLayer("regional_cuisine_restaurants",study_layer.GetSpatialRef(),ogr.wkbPoint)
    for name,typ in names:out.CreateField(ogr.FieldDefn(name,typ))
    dest.StartTransaction();count=0
    for f in layer:
        geom=f.GetGeometryRef()
        if geom is None or not boundary.Intersects(geom):continue
        item=ogr.Feature(out.GetLayerDefn());item.SetGeometry(geom)
        for name,_ in names:
            val=f.GetField(name)
            if val is not None:item.SetField(name,val)
        out.CreateFeature(item);count+=1
    dest.CommitTransaction();dest=None
    return count


def export_restaurant_auxiliary_tables(a):
    """Preserve the supplied 75-m match audit and cuisine mapping as CSV."""
    from osgeo import ogr
    ds=ogr.Open(str(a.restaurant_gdb),0)
    for layer_name in ("cross_source_matches","cuisine_to_province"):
        layer=ds.GetLayerByName(layer_name)
        if layer is None:continue
        definition=layer.GetLayerDefn();columns=[definition.GetFieldDefn(i).GetName() for i in range(definition.GetFieldCount())]
        write_csv(a.variables_dir/("restaurant_"+layer_name+".csv"),columns,[[feature.GetField(column) for column in columns] for feature in layer])


def housing_surface(a, boundary=None):
    """Local ordinary kriging of 100-m median log prices onto a clipped 200-m raster."""
    from osgeo import ogr, gdal
    path=a.variables_dir/"housing_log_price_200m.tif"
    if path.exists():return path
    study=ogr.Open(str(a.variables_dir/"study_area.gpkg"))
    if boundary is None:
        boundary_feature=study.GetLayer(0).GetNextFeature()
        boundary=boundary_feature.GetGeometryRef().Clone()
    ds=ogr.Open(str(a.variables_dir/"housing_transactions_2019.gpkg"))
    if ds is None:return None
    groups=defaultdict(list)
    for f in ds.GetLayer(0):
        value=f.GetField("price_cny_m2")
        if value and value>0:
            g=f.GetGeometryRef();groups[key(g.GetX(),g.GetY(),100,a.x_origin,a.y_origin)].append(math.log(value))
    if len(groups)<30:return None
    xy=np.array([[a.x_origin+(k[0]+.5)*100,a.y_origin+(k[1]+.5)*100] for k in groups],float)
    z=np.array([np.median(v) for v in groups.values()])
    tree=cKDTree(xy)
    x0,x1,y0,y1=boundary.GetEnvelope();left=math.floor(x0/200)*200;top=math.ceil(y1/200)*200
    nx=math.ceil((x1-left)/200);ny=math.ceil((top-y0)/200)
    array=np.full((ny,nx),-9999,dtype=np.float32)
    variance=max(float(np.var(z)),.01);range_m=1500.0;nugget=.08*variance
    def predict(point,points,values,spatial_tree):
        d,ix=spatial_tree.query(point,k=min(24,len(points)))
        if np.min(d)>5000:return np.nan
        local_xy=points[np.atleast_1d(ix)];local_values=values[np.atleast_1d(ix)]
        dd=distance.cdist(local_xy,local_xy)
        K=variance*np.exp(-.5*(dd/range_m)**2)+np.eye(len(local_xy))*nugget
        rhs=variance*np.exp(-.5*(np.atleast_1d(d)/range_m)**2)
        system=np.empty((len(local_xy)+1,len(local_xy)+1));system[:-1,:-1]=K
        system[:-1,-1]=1;system[-1,:-1]=1;system[-1,-1]=0
        try:weights=np.linalg.solve(system,np.r_[rhs,1])[:-1]
        except np.linalg.LinAlgError:return np.nan
        return float(np.dot(weights,local_values))
    random=np.random.RandomState(a.seed);order=random.permutation(len(z));holdout=order[:max(1,len(z)//5)];training=order[len(holdout):]
    train_tree=cKDTree(xy[training]);cv=np.array([predict(xy[i],xy[training],z[training],train_tree) for i in holdout])
    valid=np.isfinite(cv);actual=np.exp(z[holdout][valid]);predicted=np.exp(cv[valid])
    cv_r2=float(1-np.sum((actual-predicted)**2)/np.sum((actual-actual.mean())**2)) if len(actual)>2 and np.var(actual)>0 else None
    cv_mape=float(np.median(np.abs(predicted-actual)/actual)) if len(actual) else None
    for row in range(ny):
        cy=top-(row+.5)*200
        for col in range(nx):
            cx=left+(col+.5)*200;p=ogr.Geometry(ogr.wkbPoint);p.AddPoint_2D(cx,cy)
            if not boundary.Intersects(p):continue
            estimate=predict([cx,cy],xy,z,tree)
            if np.isfinite(estimate):array[row,col]=estimate
    driver=gdal.GetDriverByName("GTiff")
    raster=driver.Create(str(path),nx,ny,1,gdal.GDT_Float32,["COMPRESS=LZW","TILED=YES"])
    raster.SetGeoTransform((left,200,0,top,0,-200))
    raster.SetProjection(study.GetLayer(0).GetSpatialRef().ExportToWkt())
    band=raster.GetRasterBand(1);band.SetNoDataValue(-9999);band.WriteArray(array)
    raster=None
    (a.variables_dir/"housing_kriging_parameters.json").write_text(json.dumps({"input_2019_transactions":sum(map(len,groups.values())),"aggregated_100m_locations":len(groups),"grid_resolution_m":200,"local_neighbours":24,"gaussian_covariance_range_m":range_m,"nugget_fraction_of_sample_variance":.08,"holdout_fraction":.2,"holdout_r2_raw_price":cv_r2,"holdout_median_absolute_percentage_error":cv_mape,"note":"Reconstructed local ordinary kriging; original fitted variogram and holdout split were not supplied."},indent=2),encoding="utf-8")
    return path


def read_context(a, grid_size=400, required_keys=()):
    """Return cell-level urban covariates from the clipped variable folder."""
    from osgeo import ogr,gdal
    output=defaultdict(lambda:defaultdict(float));shopping_xy=[]
    for layer_name,out_name in [("non_restaurant_pois","non_restaurant_pois"),("shopping_pois","shopping_pois"),("building_footprints","building_count")]:
        ds=ogr.Open(str(a.variables_dir/(layer_name+".gpkg")))
        if ds is None:continue
        for f in ds.GetLayer(0):
            g=f.GetGeometryRef()
            if g is None:continue
            if layer_name=="building_footprints":g=g.Centroid()
            if layer_name=="shopping_pois":shopping_xy.append((g.GetX(),g.GetY()))
            output[key(g.GetX(),g.GetY(),grid_size,a.x_origin,a.y_origin)][out_name]+=1
    ds=ogr.Open(str(a.variables_dir/"roads.gpkg"))
    if ds:
        for f in ds.GetLayer(0):
            geom=f.GetGeometryRef()
            if geom is None:continue
            xmin,xmax,ymin,ymax=geom.GetEnvelope()
            for gx in range(math.floor((xmin-a.x_origin)/grid_size),math.floor((xmax-a.x_origin)/grid_size)+1):
                for gy in range(math.floor((ymin-a.y_origin)/grid_size),math.floor((ymax-a.y_origin)/grid_size)+1):
                    square=ogr.CreateGeometryFromWkt("POLYGON (({0} {1},{2} {1},{2} {3},{0} {3},{0} {1}))".format(a.x_origin+gx*grid_size,a.y_origin+gy*grid_size,a.x_origin+(gx+1)*grid_size,a.y_origin+(gy+1)*grid_size))
                    if geom.Intersects(square):
                        cut=geom.Intersection(square)
                        if cut:output[(gx,gy)]["road_density_km_per_km2"]+=cut.Length()*1000/(grid_size**2)
    metro=[];ds=ogr.Open(str(a.variables_dir/"metro_stations.gpkg"))
    if ds:
        for f in ds.GetLayer(0):
            g=f.GetGeometryRef();metro.append((g.GetX(),g.GetY()))
    metro_tree=cKDTree(np.asarray(metro)) if metro else None
    shopping_tree=cKDTree(np.asarray(shopping_xy)) if shopping_xy else None
    cbd_ds=ogr.Open(str(a.variables_dir/"cbd_reference_point.gpkg"))
    if cbd_ds:
        cbd_feature=cbd_ds.GetLayer(0).GetNextFeature()
        cbd_g=cbd_feature.GetGeometryRef().Clone();cbd_xy=(cbd_g.GetX(),cbd_g.GetY())
    else:
        pop_ds,pop_layer=gdb_layer(a.population_gdb,"population_points")
        from osgeo import osr
        s=osr.SpatialReference();s.ImportFromEPSG(4326);g=ogr.Geometry(ogr.wkbPoint);g.AddPoint_2D(a.cbd_longitude,a.cbd_latitude)
        g.Transform(osr.CoordinateTransformation(s,pop_layer.GetSpatialRef()));cbd_xy=(g.GetX(),g.GetY())
    for layer_name,name in [("nighttime_light_100m.tif","nighttime_light"),("housing_log_price_200m.tif","housing_log_price")]:
        if name=="housing_log_price" and not (a.variables_dir/layer_name).exists():housing_surface(a)
        raster=gdal.Open(str(a.variables_dir/layer_name))
        if raster is None:continue
        arr=raster.GetRasterBand(1).ReadAsArray();gt=raster.GetGeoTransform();nodata=raster.GetRasterBand(1).GetNoDataValue()
        sums=defaultdict(float);counts=defaultdict(int)
        for row in range(arr.shape[0]):
            cy=gt[3]+(row+.5)*gt[5]
            for col in range(arr.shape[1]):
                v=float(arr[row,col]);
                if not np.isfinite(v) or (nodata is not None and v==nodata):continue
                cx=gt[0]+(col+.5)*gt[1];k=key(cx,cy,grid_size,a.x_origin,a.y_origin)
                sums[k]+=v;counts[k]+=1
        for k,total in sums.items():output[k][name]=total/counts[k]
    for k in required_keys:output[k]
    for k,row in output.items():
        cx=a.x_origin+(k[0]+.5)*grid_size;cy=a.y_origin+(k[1]+.5)*grid_size
        row["cbd_distance_km"]=math.hypot(cx-cbd_xy[0],cy-cbd_xy[1])/1000
        row["metro_distance_km"]=float(metro_tree.query([cx,cy])[0])/1000 if metro_tree else np.nan
        # Shopping points are the user-specified source for commercial context.
        # Count within 250 m of the centroid is a proxy, not a parcel-area measure.
        row["commercial_area_proxy"]=len(shopping_tree.query_ball_point([cx,cy],r=250)) if shopping_tree else 0
    return output


def export_analysis_grids(a):
    """Export the ten aligned square-grid geometries, clipped to the study polygon."""
    from osgeo import ogr
    study=ogr.Open(str(a.variables_dir/"study_area.gpkg"));layer=study.GetLayer(0)
    boundary_feature=layer.GetNextFeature();boundary=boundary_feature.GetGeometryRef().Clone();srs=layer.GetSpatialRef()
    xmin,xmax,ymin,ymax=boundary.GetEnvelope();driver=ogr.GetDriverByName("GPKG")
    for size in SCALES:
        dest_path=a.variables_dir/f"analysis_grid_{size}m.gpkg"
        if dest_path.exists():continue
        ds=driver.CreateDataSource(str(dest_path));out=ds.CreateLayer(f"analysis_grid_{size}m",srs,ogr.wkbUnknown)
        for name,typ in [("grid_x",ogr.OFTInteger),("grid_y",ogr.OFTInteger),("clipped_area_m2",ogr.OFTReal)]:out.CreateField(ogr.FieldDefn(name,typ))
        ds.StartTransaction();count=0
        for gx in range(math.floor((xmin-a.x_origin)/size),math.floor((xmax-a.x_origin)/size)+1):
            for gy in range(math.floor((ymin-a.y_origin)/size),math.floor((ymax-a.y_origin)/size)+1):
                x0=a.x_origin+gx*size;y0=a.y_origin+gy*size
                square=ogr.CreateGeometryFromWkt("POLYGON (({0} {1},{2} {1},{2} {3},{0} {3},{0} {1}))".format(x0,y0,x0+size,y0+size))
                if not boundary.Intersects(square):continue
                clipped=square if boundary.Contains(square) else square.Intersection(boundary)
                if clipped is None or clipped.IsEmpty() or clipped.GetArea()==0:continue
                f=ogr.Feature(out.GetLayerDefn());f.SetGeometry(clipped)
                f.SetField("grid_x",gx);f.SetField("grid_y",gy);f.SetField("clipped_area_m2",clipped.GetArea())
                out.CreateFeature(f);count+=1
                if count%25000==0:ds.CommitTransaction();ds.StartTransaction()
        ds.CommitTransaction();ds=None
        print("Exported",dest_path.name,count,flush=True)


def key(x, y, size, x0, y0):
    return (math.floor((x - x0) / size), math.floor((y - y0) / size))


def empty_cell():
    return {"p": np.zeros(31, dtype=np.int64), "m": np.zeros(31, dtype=np.int64), "r": np.zeros(31, dtype=np.int64), "c": np.zeros(16, dtype=np.int64), "age_sum": 0, "age_n": 0, "male": 0, "sex_n": 0}


def read_points(a):
    cells = defaultdict(empty_cell)
    summary = {"population_total": 0, "population_eligible": 0, "restaurants_total": 0, "restaurants_candidate": 0}
    ds, layer = gdb_layer(a.population_gdb, "population_points")
    for f in layer:
        summary["population_total"] += 1
        p = ZH_P.get(f.GetField("hukou_prov"))
        typ = f.GetField("pop_type")
        if p is None or typ not in ("migrant_population", "resident_population"):
            continue
        geo = f.GetGeometryRef()
        if geo is None:
            continue
        cell = cells[key(geo.GetX(), geo.GetY(), 100, a.x_origin, a.y_origin)]
        cell["p"][p] += 1
        cell["m" if typ == "migrant_population" else "r"][p] += 1
        age = f.GetField("age2020")
        if isinstance(age, int) and 0 <= age <= 110:
            cell["age_sum"] += age
            cell["age_n"] += 1
        sex = f.GetField("sex")
        if sex in (1, 2):
            cell["sex_n"] += 1
            cell["male"] += int(sex == 1)  # Verify the coding against the data custodian.
        summary["population_eligible"] += 1
        if a.limit_pop and summary["population_total"] >= a.limit_pop:
            break
    del layer, ds
    restaurant_points = []
    if a.restaurant_gdb.exists():
        ds,layer=gdb_layer(a.restaurant_gdb,"restaurant_points")
    else:
        from osgeo import ogr
        ds=ogr.Open(str(a.restaurant_vector.resolve()),0)
        if ds is None:raise FileNotFoundError("Restaurant GDB and clipped restaurant vector are both missing")
        layer=ds.GetLayerByName("regional_cuisine_restaurants")
        if layer is None:raise ValueError("regional_cuisine_restaurants layer missing in "+str(a.restaurant_vector))
    for f in layer:
        summary["restaurants_total"] += 1
        c = ZH_C.get(f.GetField("cuisine"))
        geo = f.GetGeometryRef()
        if c is None or geo is None:
            continue
        k = key(geo.GetX(), geo.GetY(), 100, a.x_origin, a.y_origin)
        cells[k]["c"][c] += 1
        restaurant_points.append((geo.GetX(), geo.GetY(), c, f.GetField("source"), f.GetField("method")))
        summary["restaurants_candidate"] += 1
    del layer, ds
    return cells, restaurant_points, summary


def aggregate(base, scale):
    if scale == 100:
        return base
    out = defaultdict(empty_cell)
    for (gx, gy), cell in base.items():
        dest = out[(math.floor(gx * 100 / scale), math.floor(gy * 100 / scale))]
        for name in ("p", "m", "r", "c"):
            dest[name] += cell[name]
        for name in ("age_sum", "age_n", "male", "sex_n"):
            dest[name] += cell[name]
    return out


def frame(cells, scale, a):
    keys = list(cells)
    p = np.stack([cells[k]["p"] for k in keys])
    c = np.stack([cells[k]["c"] for k in keys])
    m = np.stack([cells[k]["m"] for k in keys])
    r = np.stack([cells[k]["r"] for k in keys])
    n, t = p.sum(1), c.sum(1)
    xy = np.array([[a.x_origin + (k[0] + .5) * scale, a.y_origin + (k[1] + .5) * scale] for k in keys])
    return {"keys": keys, "xy": xy, "p": p, "c": c, "m": m, "r": r, "n": n, "t": t, "joint": (n >= 30) & (t >= 3), "panel": (n >= 30) & (t >= 1), "cells": cells}


def entropy(counts):
    totals = counts.sum(1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
    h = -(np.where(shares > 0, shares * np.log(np.maximum(shares, 1e-15)), 0)).sum(1)
    simpson = 1 - (shares ** 2).sum(1)
    richness = (counts > 0).sum(1)
    return richness, h, simpson


def neighbors(xy, k=8):
    if len(xy) <= k:
        return None
    return cKDTree(xy).query(xy, k=k+1)[1][:, 1:]


def moran(x, nn, y=None):
    if nn is None or len(x) < 3:
        return np.nan
    x=np.asarray(x,float);zx=x-x.mean()
    if y is None:
        den=np.sum(zx*zx)
        return float(np.sum(zx*zx[nn].mean(1))/den) if den>0 else np.nan
    y=np.asarray(y,float);zy=y-y.mean()
    sx=zx.std();sy=zy.std()
    return float(np.mean((zx/sx)*(zy[nn]/sy).mean(1))) if sx>0 and sy>0 else np.nan


def direct_stats(f):
    sel = f["panel"]
    p, c = f["p"][sel].astype(float), f["c"][sel].astype(float)
    share = p / p.sum(1)[:, None]
    t = c.sum(1)
    baseline = (share * t[:, None]).sum(0) / t.sum()
    exposure = (share.T @ c) / np.maximum(c.sum(0), 1)[None, :]
    matrix = exposure / np.maximum(baseline[:, None], 1e-15)
    matched = share @ MATCH
    base_match = baseline @ MATCH
    direct = (matched * c).sum(0) / np.maximum(c.sum(0), 1) / np.maximum(base_match, 1e-15)
    local = np.sum((c / t[:, None]) * (matched / np.maximum(base_match, 1e-15)), axis=1)
    joint_within_panel=f["joint"][sel]
    return {"baseline": baseline, "matrix": matrix, "direct": direct, "local": local[joint_within_panel], "local_sel":f["joint"], "sel": sel, "c": c, "share": share, "t": t, "matched": matched}


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p); q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    return np.minimum(q, 1)


def permute_direct(st, repetitions, rng):
    # Shuffling labels across individual restaurant locations preserves cuisine totals.
    c = st["c"].astype(int)
    cell_idx, labels = [], []
    for i in range(len(c)):
        for j, n in enumerate(c[i]):
            if n:
                cell_idx.extend([i] * int(n)); labels.extend([j] * int(n))
    cell_idx = np.asarray(cell_idx, dtype=int); labels = np.asarray(labels, dtype=int)
    num_c = c.sum(0)
    matched = st["matched"]
    baseline_match = st["baseline"] @ MATCH
    observed = st["direct"]
    ge = np.zeros(16, dtype=int); eq = 0; weighted = 0
    observed_eq = observed.mean(); observed_weighted = np.average(observed, weights=num_c)
    for _ in range(repetitions):
        perm = rng.permutation(labels)
        cp = np.bincount(cell_idx * 16 + perm, minlength=len(c)*16).reshape(-1, 16)
        q = (matched * cp).sum(0) / num_c / np.maximum(baseline_match, 1e-15)
        ge += q >= observed
        eq += q.mean() >= observed_eq
        weighted += np.average(q, weights=num_c) >= observed_weighted
    return {"direct_p": (ge+1)/(repetitions+1), "direct_q": bh((ge+1)/(repetitions+1)), "equal_p": (eq+1)/(repetitions+1), "weighted_p": (weighted+1)/(repetitions+1)}


def poisson_fit(y, xvars, offset, grid=None, cuisine=None, cluster=None):
    y = np.asarray(y, float); xvars = np.asarray(xvars, float)
    n, k = xvars.shape
    scales=np.std(xvars,axis=0)
    scales=np.where(scales>1e-8,scales,1.0)
    mats = [sparse.csr_matrix(xvars/scales)]
    if grid is None:
        mats.insert(0, sparse.csr_matrix(np.ones((n, 1))))
        coef_offset = 1
    else:
        ng = int(np.max(grid)) + 1
        mats.append(sparse.csr_matrix((np.ones(n), (np.arange(n), grid)), shape=(n, ng)))
        coef_offset = 0
    if cuisine is not None:
        nc = int(np.max(cuisine)) + 1
        keep = cuisine < nc-1
        mats.append(sparse.csr_matrix((np.ones(keep.sum()), (np.where(keep)[0], cuisine[keep])), shape=(n, nc-1)))
    X = sparse.hstack(mats, format="csr")
    start = np.zeros(X.shape[1])
    initial_intercept=np.log((y.sum()+1)/(np.exp(offset).sum()+1))
    if grid is None:start[0]=initial_intercept
    else:start[k:k+ng]=initial_intercept
    def objective(theta):
        eta = np.clip(offset + X @ theta, -30, 30)
        mu = np.exp(eta)
        return float((mu-y*eta).sum()), np.asarray(X.T @ (mu-y)).ravel()
    res = minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 4000, "ftol": 1e-10, "gtol": 1e-6,"maxls":50})
    eta = np.clip(offset + X @ res.x, -30, 30); mu = np.exp(eta)
    H = (X.T @ X.multiply(mu[:, None])).tocsc()
    groups = np.arange(n) if cluster is None else cluster
    estimates = []
    for j in range(k):
        e = np.zeros(X.shape[1]);e[j+coef_offset] = 1
        v = spsolve(H + sparse.eye(H.shape[0], format="csc")*1e-9, e)
        influence = np.asarray(X @ v).ravel() * (y-mu)
        score = np.bincount(groups, weights=influence)
        se = np.sqrt(np.sum(score**2))/scales[j]
        beta=res.x[j+coef_offset]/scales[j]
        estimates.append({"beta": float(beta), "se_cluster": float(se), "irr": float(np.exp(beta)), "ci_low": float(np.exp(beta-1.96*se)), "ci_high": float(np.exp(beta+1.96*se))})
    return {"coef": estimates, "converged": bool(res.success), "message": str(res.message), "pred": mu}


def regression(f,context=None):
    sel = f["panel"];p=f["p"][sel].astype(float);c=f["c"][sel].astype(float)
    shares=p/p.sum(1)[:,None];matched=shares@MATCH
    y=c.ravel();x=(matched/0.1).ravel()[:,None]
    g=np.repeat(np.arange(len(c)),16);cu=np.tile(np.arange(16),len(c));off=np.repeat(np.log(c.sum(1)),16)
    pooled=poisson_fit(y,x,off,cuisine=cu,cluster=g)
    source=[f["cells"][f["keys"][i]] for i in np.where(sel)[0]]
    cov=np.column_stack([
        np.log(p.sum(1)),
        np.array([z["m"].sum() for z in source])/p.sum(1),
        entropy(p)[1],
        np.array([z["age_sum"]/z["age_n"] if z["age_n"] else np.nan for z in source]),
        np.array([z["male"]/z["sex_n"] if z["sex_n"] else np.nan for z in source]),
    ])
    demographic=None
    if np.isfinite(cov).all():
        demographic=poisson_fit(y,np.column_stack([x,np.repeat(cov,16,axis=0)]),off,cuisine=cu,cluster=g)
    urban=None;urban_names=[];urban_data=None
    if context is not None and demographic is not None:
        urban_names=["non_restaurant_pois","shopping_pois","cbd_distance_km","metro_distance_km","road_density_km_per_km2","building_count","commercial_area_proxy","nighttime_light"]
        urban_data=np.array([[context[f["keys"][i]].get(name,np.nan) for name in urban_names] for i in np.where(sel)[0]],float)
        for j in range(urban_data.shape[1]):
            if urban_names[j] in ("non_restaurant_pois","shopping_pois","building_count","commercial_area_proxy","nighttime_light"):
                urban_data[:,j]=np.log1p(np.maximum(urban_data[:,j],0))
            finite=np.isfinite(urban_data[:,j]);median=np.median(urban_data[finite,j]) if finite.any() else 0
            urban_data[~finite,j]=median
            sd=np.std(urban_data[:,j]);urban_data[:,j]=(urban_data[:,j]-np.mean(urban_data[:,j]))/(sd if sd>0 else 1)
        urban=poisson_fit(y,np.column_stack([x,np.repeat(cov,16,axis=0),np.repeat(urban_data,16,axis=0)]),off,cuisine=cu,cluster=g)
    fixed=poisson_fit(y,x,off,grid=g,cuisine=cu,cluster=g)
    return pooled,demographic,urban,fixed,{"y":y,"x":x,"grid":g,"cuisine":cu,"offset":off,"cell_indices":np.where(sel)[0],"urban_names":urban_names,"urban_standardized":urban_data}


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer=csv.writer(fh);writer.writerow(header);writer.writerows(rows)


def save_scale(f, st, scale, out, a):
    keys=f["keys"];p=f["p"];c=f["c"];n=f["n"];t=f["t"]
    hp,h,hs=entropy(p);cp,ch,cs=entropy(c)
    rows=[]
    for i,(gx,gy) in enumerate(keys):
        cell=f["cells"][keys[i]]
        rows.append([gx,gy,a.x_origin+(gx+.5)*scale,a.y_origin+(gy+.5)*scale,int(n[i]),int(t[i]),int(f["joint"][i]),int(f["panel"][i]),int(hp[i]),float(h[i]),float(hs[i]),int(cp[i]),float(ch[i]),float(cs[i]),int(cell["m"].sum()),int(cell["r"].sum()),float(cell["age_sum"]/cell["age_n"]) if cell["age_n"] else "",float(cell["male"]/cell["sex_n"]) if cell["sex_n"] else "",*p[i],*c[i]])
    write_csv(out/f"grid_{scale:04d}m.csv",["grid_x","grid_y","centre_x_m","centre_y_m","eligible_residents","candidate_restaurants","joint_eligible","model_eligible","origin_richness","origin_shannon","origin_gini_simpson","cuisine_richness","cuisine_shannon","cuisine_gini_simpson","migrant_count","resident_count","mean_age","male_share_assuming_1_is_male",*["hukou_"+s for s in PROVINCES],*["cuisine_"+s for s in CUISINES]],rows)
    both=f["joint"]
    popsel=n>=30; cuisinesel=t>=3
    nn_joint=neighbors(f["xy"][both])
    nn_pop=neighbors(f["xy"][popsel])
    nn_cuisine=neighbors(f["xy"][cuisinesel])
    return {"scale_m":scale,"populated_cells":int((n>0).sum()),"joint_cells":int(both.sum()),"model_cells":int(f["panel"].sum()),"population_coverage":float(n[both].sum()/max(n.sum(),1)),"restaurant_coverage":float(t[both].sum()/max(t.sum(),1)),"spearman_shannon":float(stats.spearmanr(h[both],ch[both]).correlation) if both.sum()>3 else None,"moran_origin":moran(h[popsel],nn_pop),"moran_cuisine":moran(ch[cuisinesel],nn_cuisine),"bivariate_moran":moran(h[both],nn_joint,ch[both]),"equal_direct":float(st["direct"].mean()),"weighted_direct":float(np.average(st["direct"],weights=st["c"].sum(0))),"median_local":float(np.median(st["local"])),"local_q1":float(np.quantile(st["local"],.25)),"local_q3":float(np.quantile(st["local"],.75)),"local_share_above_one":float(np.mean(st["local"]>1)),"local_log_moran":moran(np.log(np.maximum(st["local"],1e-12)),nn_joint),"mean_origin_richness":float(hp[both].mean()),"mean_cuisine_richness":float(cp[both].mean()),"mean_origin_shannon":float(h[both].mean()),"mean_cuisine_shannon":float(ch[both].mean()),"mean_origin_simpson":float(hs[both].mean()),"mean_cuisine_simpson":float(cs[both].mean())}


def local_bivariate(f, iterations, rng):
    sel=f["joint"];p=f["p"][sel];c=f["c"][sel]
    x=entropy(p)[1];y=entropy(c)[1];xy=f["xy"][sel]
    nn=neighbors(xy)
    if nn is None:
        return None
    zx=(x-x.mean())/max(x.std(),1e-12)
    zy=(y-y.mean())/max(y.std(),1e-12)
    lag=zy[nn].mean(1);obs=zx*lag
    extreme=np.zeros(len(obs),dtype=int)
    for _ in range(iterations):
        perm=rng.permutation(zy)
        extreme+=(np.abs(zx*perm[nn].mean(1))>=np.abs(obs))
    pval=(extreme+1)/(iterations+1)
    return {"xy":xy,"local_i":obs,"p":pval,"q":bh(pval),"zx":zx,"lag_zy":lag}


def plot_additional(scales,detail,model,panel,out,rng,iterations):
    f=detail[400]["frame"]
    fig,axs=plt.subplots(1,2,figsize=(13,5),dpi=180)
    p=f["p"].sum(0);c=f["c"].sum(0)
    axs[0].bar(range(31),p/p.sum());axs[0].set_xticks(range(31));axs[0].set_xticklabels(PROVINCES,rotation=90,fontsize=6);axs[0].set_title("Citywide hukou-origin shares")
    axs[1].bar(range(16),c/c.sum());axs[1].set_xticks(range(16));axs[1].set_xticklabels(CUISINES,rotation=90,fontsize=7);axs[1].set_title("Citywide candidate-cuisine shares")
    fig.tight_layout();fig.savefig(out/"citywide_compositions.png");plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),dpi=180)
    for ax,metric,title in zip(axs,["richness","shannon","simpson"],["Richness","Shannon entropy","Gini–Simpson"]):
        for prefix,label in [("origin","Hukou origin"),("cuisine","Cuisine")]:
            ax.plot([z["scale_m"] for z in scales],[z["mean_"+prefix+"_"+metric] for z in scales],marker="o",label=label)
        ax.set_title(title);ax.set_xlabel("Grid size (m)");ax.legend()
    fig.tight_layout();fig.savefig(out/"supp_S1_diversity_scales.png");plt.close(fig)
    for size in (300,400,500):
        ff=detail[size]["frame"];sel=ff["joint"]
        x=entropy(ff["p"][sel])[1];y=entropy(ff["c"][sel])[1]
        fig,ax=plt.subplots(figsize=(5,5),dpi=180);ax.hexbin(x,y,gridsize=30,mincnt=1,cmap="viridis")
        ax.set_xlabel("Hukou Shannon entropy");ax.set_ylabel("Cuisine Shannon entropy");ax.set_title(f"Same-cell diversity, {size} m")
        fig.tight_layout();fig.savefig(out/f"supp_S4_diversity_pairing_{size}m.png");plt.close(fig)
        local=local_bivariate(ff,iterations,rng)
        if local is None:continue
        write_csv(out/f"supp_S3_local_bivariate_{size}m.csv",["centre_x_m","centre_y_m","local_i","nominal_p","BH_q"],zip(local["xy"][:,0],local["xy"][:,1],local["local_i"],local["p"],local["q"]))
        fig,ax=plt.subplots(figsize=(6,6),dpi=180)
        colour=np.where(local["p"]>=.05,"#d9d9d9",np.where(local["zx"]>0,np.where(local["lag_zy"]>0,"#c33c54","#f3a54a"),np.where(local["lag_zy"]>0,"#6fb8dc","#315c91")))
        ax.scatter(local["xy"][:,0],local["xy"][:,1],c=colour,s=8,marker="s")
        ax.set_aspect("equal");ax.set_title(f"Nominal local bivariate patterns, {size} m")
        fig.tight_layout();fig.savefig(out/f"supp_S3_local_bivariate_{size}m.png");plt.close(fig)
    y=panel["y"];mu=model[3]["pred"]
    fig,axs=plt.subplots(1,2,figsize=(10,4),dpi=180)
    axs[0].hexbin(y,mu,gridsize=40,mincnt=1,cmap="viridis");axs[0].plot([0,max(y.max(),mu.max())],[0,max(y.max(),mu.max())],"k--",lw=.7)
    axs[0].set_xlabel("Observed count");axs[0].set_ylabel("Predicted count")
    residual=(y-mu)/np.sqrt(np.maximum(mu,1e-10));axs[1].hist(residual,bins=80)
    axs[1].set_xlabel("Pearson residual");fig.tight_layout();fig.savefig(out/"supp_S7_model_diagnostics.png");plt.close(fig)
    residual_cell=np.bincount(panel["grid"],weights=np.abs(residual))/16
    xy=f["xy"][panel["cell_indices"]]
    fig,ax=plt.subplots(figsize=(6,6),dpi=180)
    im=ax.scatter(xy[:,0],xy[:,1],c=residual_cell,s=9,marker="s",cmap="magma")
    ax.set_aspect("equal");ax.set_title("Mean absolute Pearson residual, 400 m")
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/"supp_S7_spatial_residuals.png");plt.close(fig)


def matrix_permutation(st,repetitions,rng):
    c=st["c"].astype(int);share=st["share"]
    cell_idx=np.repeat(np.arange(len(c)),c.sum(1));labels=np.concatenate([np.repeat(np.arange(16),row) for row in c])
    total=c.sum(0);baseline=st["baseline"]
    observed=st["matrix"];extreme=np.zeros(observed.shape,dtype=int)
    for _ in range(repetitions):
        labels_perm=rng.permutation(labels)
        cp=np.bincount(cell_idx*16+labels_perm,minlength=len(c)*16).reshape(-1,16)
        quotient=(share.T@cp)/np.maximum(total,1)[None,:]/np.maximum(baseline[:,None],1e-15)
        extreme+=np.abs(quotient-1)>=np.abs(observed-1)
    p=(extreme+1)/(repetitions+1)
    return p,bh(p.ravel()).reshape(p.shape)


def variogram_and_thresholds(detail,out,rng):
    records=[];threshold=[]
    for scale,st in detail.items():
        f=st["frame"]
        for domain,counts,minimum in [("origin",f["p"],f["n"]>=30),("cuisine",f["c"],f["t"]>=3)]:
            xy=f["xy"][minimum];value=entropy(counts[minimum])[1]
            if len(value)<30:continue
            if len(value)>1200:
                choose=rng.choice(len(value),1200,replace=False);xy=xy[choose];value=value[choose]
            value=(value-value.mean())/max(value.std(),1e-12)
            dd=distance.pdist(xy);semi=.5*distance.pdist(value[:,None],"sqeuclidean")
            bins=np.arange(0,12500,500);bin_id=np.digitize(dd,bins)-1
            mean_d=[];mean_y=[];num=[]
            for i in range(len(bins)-1):
                mask=bin_id==i
                if mask.sum()>=30:
                    mean_d.append(float(dd[mask].mean()));mean_y.append(float(semi[mask].mean()));num.append(int(mask.sum()))
            ratio=np.nan
            if len(mean_d)>=5:
                def gaussian(x,nugget,sill,range_m):return nugget+sill*(1-np.exp(-.5*(x/range_m)**2))
                try:
                    fit,_=curve_fit(gaussian,mean_d,mean_y,p0=[.1,1,3000],bounds=([0,.01,100],[2,5,20000]),maxfev=5000)
                    ratio=float(fit[0]/(fit[0]+fit[1]))
                except (RuntimeError,ValueError):pass
            records.append((scale,domain,len(value),ratio))
            write_csv(out/f"supp_S2_variogram_{domain}_{scale}m.csv",["distance_m","semivariance","pairs"],zip(mean_d,mean_y,num))
        for minimum_rest in (1,3,5,10):
            use=(f["n"]>=30)&(f["t"]>=minimum_rest)
            h=entropy(f["p"][use])[1];ch=entropy(f["c"][use])[1]
            rho=float(stats.spearmanr(h,ch).correlation) if len(h)>3 else np.nan
            cross=moran(h,neighbors(f["xy"][use]),ch)
            threshold.append((scale,minimum_rest,int(use.sum()),rho,cross))
    write_csv(out/"supp_S2_variogram_summary.csv",["scale_m","domain","sampled_cells","nugget_to_sill"],records)
    write_csv(out/"supp_S2_restaurant_thresholds.csv",["scale_m","minimum_restaurants","cells","spearman","bivariate_moran"],threshold)
    fig,axs=plt.subplots(1,2,figsize=(11,4),dpi=180)
    for domain in ("origin","cuisine"):
        rr=[r for r in records if r[1]==domain];axs[0].plot([r[0] for r in rr],[r[3] for r in rr],marker="o",label=domain)
    axs[0].set_title("Gaussian variogram nugget/sill");axs[0].legend()
    for minimum_rest in (1,3,5,10):
        rr=[r for r in threshold if r[1]==minimum_rest];axs[1].plot([r[0] for r in rr],[r[3] for r in rr],marker="o",label=str(minimum_rest))
    axs[1].set_title("Same-cell Shannon association");axs[1].legend(title="Min restaurants")
    for ax in axs:ax.set_xlabel("Grid size (m)")
    fig.tight_layout();fig.savefig(out/"supp_S2_variogram_and_thresholds.png");plt.close(fig)


def source_sensitivity(f,restaurant_points,a,out):
    result=[];keys=f["keys"];index={k:i for i,k in enumerate(keys)}
    variants=[("full",lambda r:True,list(range(16))),
              ("baidu_only",lambda r:r[3]=="baidu",list(range(16))),
              ("dianping_only",lambda r:r[3]=="dianping",list(range(16))),
              ("direct_category_only",lambda r:r[4]==1,list(range(16))),
              ("name_or_dish_only",lambda r:r[4]==2,list(range(16))),
              ("province_specific_only",lambda r:r[2]<13,list(range(13))),
              ("exclude_guangdong",lambda r:r[2]!=0,list(range(1,16)))]
    for name,keep,selected in variants:
        c=np.zeros_like(f["c"])
        for r in restaurant_points:
            if not keep(r):continue
            i=index.get(key(r[0],r[1],400,a.x_origin,a.y_origin))
            if i is not None:c[i,r[2]]+=1
        alternative=dict(f);alternative["c"]=c;alternative["t"]=c.sum(1);alternative["panel"]=(f["n"]>=30)&(alternative["t"]>=1)
        s=direct_stats(alternative)
        total=s["c"].sum(0);good=[j for j in selected if total[j]>0]
        result.append((name,int(alternative["panel"].sum()),int(total.sum()),float(np.mean(s["direct"][good])),float(np.average(s["direct"][good],weights=total[good]))))
    write_csv(out/"supp_S6_source_classification_sensitivity.csv",["variant","model_cells","restaurants","equal_cuisine_quotient","restaurant_weighted_quotient"],result)
    fig,ax=plt.subplots(figsize=(8,4),dpi=180)
    x=np.arange(len(result));ax.plot(x,[r[3] for r in result],"o-",label="Equal cuisine");ax.plot(x,[r[4] for r in result],"s-",label="Restaurant weighted")
    ax.axhline(1,color="grey",linestyle="--");ax.set_xticks(x);ax.set_xticklabels([r[0] for r in result],rotation=35,ha="right")
    ax.legend();fig.tight_layout();fig.savefig(out/"supp_S6_source_sensitivity.png");plt.close(fig)
    return result


def model_extensions(detail,context,out):
    """Adjacent scales, population types, contextual interactions and price strata."""
    rows=[]
    for size in (300,500):
        f=detail[size]["frame"];fit=regression(f)[3]
        z=fit["coef"][0];rows.append(("scale_{}m".format(size),z["irr"],z["ci_low"],z["ci_high"],int(f["panel"].sum()),fit["converged"]))
    f=detail[400]["frame"];sel=f["panel"]
    m=f["m"][sel].astype(float);r=f["r"][sel].astype(float);c=f["c"][sel]
    valid=(m.sum(1)>=10)&(r.sum(1)>=10)
    c=c[valid];m=m[valid];r=r[valid]
    if len(c)>10:
        mshare=(m/m.sum(1)[:,None])@MATCH;rshare=(r/r.sum(1)[:,None])@MATCH
        grid=np.repeat(np.arange(len(c)),16);cu=np.tile(np.arange(16),len(c));off=np.repeat(np.log(c.sum(1)),16)
        x=np.column_stack([(rshare/.1).ravel(),(mshare/.1).ravel()])
        fit=poisson_fit(c.ravel(),x,off,grid=grid,cuisine=cu,cluster=grid)
        for name,z in zip(("resident_population","migrant_population"),fit["coef"]):
            rows.append((name,z["irr"],z["ci_low"],z["ci_high"],len(c),fit["converged"]))
    panel=f["panel"];p=f["p"][panel].astype(float);c=f["c"][panel]
    matched=(p/p.sum(1)[:,None])@MATCH
    cell_keys=[f["keys"][i] for i in np.where(panel)[0]]
    urban={"cbd_distance_km":np.array([context[k].get("cbd_distance_km",np.nan) for k in cell_keys]),
           "shopping_pois":np.array([context[k].get("shopping_pois",0) for k in cell_keys]),
           "migrant_share":f["m"][panel].sum(1)/p.sum(1),
           "housing_log_price":np.array([context[k].get("housing_log_price",np.nan) for k in cell_keys])}
    price=urban["housing_log_price"];has_price=np.isfinite(price)
    if has_price.sum()>100:
        panel_indices=np.where(panel)[0]
        demographic=[]
        for local_i,global_i in enumerate(panel_indices):
            cell=f["cells"][f["keys"][global_i]]
            demographic.append([math.log(p[local_i].sum()),urban["migrant_share"][local_i],entropy(p[local_i:local_i+1])[1][0],cell["age_sum"]/cell["age_n"] if cell["age_n"] else np.nan,cell["male"]/cell["sex_n"] if cell["sex_n"] else np.nan])
        demographic=np.asarray(demographic,float)
        urban_names=["non_restaurant_pois","shopping_pois","cbd_distance_km","metro_distance_km","road_density_km_per_km2","building_count","commercial_area_proxy","nighttime_light"]
        urban_values=np.array([[context[k].get(name,np.nan) for name in urban_names] for k in cell_keys],float)
        for j,name in enumerate(urban_names):
            if name in ("non_restaurant_pois","shopping_pois","building_count","commercial_area_proxy","nighttime_light"):
                urban_values[:,j]=np.log1p(np.maximum(urban_values[:,j],0))
        controls=np.column_stack([demographic,urban_values,price])[has_price]
        for j in range(controls.shape[1]):
            finite=np.isfinite(controls[:,j]);controls[~finite,j]=np.median(controls[finite,j]) if finite.any() else 0
            sd=controls[:,j].std();controls[:,j]=(controls[:,j]-controls[:,j].mean())/(sd if sd>0 else 1)
        cc=c[has_price];mm=matched[has_price]
        x=np.column_stack([(mm/.1).ravel(),np.repeat(controls,16,axis=0)])
        g=np.repeat(np.arange(len(cc)),16);cu=np.tile(np.arange(16),len(cc));off=np.repeat(np.log(cc.sum(1)),16)
        fit=poisson_fit(cc.ravel(),x,off,cuisine=cu,cluster=g)
        z=fit["coef"][0];rows.append(("housing_adjusted_pooled",z["irr"],z["ci_low"],z["ci_high"],len(cc),fit["converged"]))
    for name,v in urban.items():
        finite=np.isfinite(v)
        if finite.sum()<30:continue
        v=v.copy();v[~finite]=np.median(v[finite]);v=(v-v.mean())/max(v.std(),1e-12)
        x=np.column_stack([(matched/.1).ravel(),((matched/.1)*v[:,None]).ravel()])
        g=np.repeat(np.arange(len(c)),16);cu=np.tile(np.arange(16),len(c));off=np.repeat(np.log(c.sum(1)),16)
        fit=poisson_fit(c.ravel(),x,off,grid=g,cuisine=cu,cluster=g)
        z=fit["coef"][1];rows.append(("interaction_"+name,z["irr"],z["ci_low"],z["ci_high"],len(c),fit["converged"]))
    finite=np.isfinite(price)
    quintile=np.full(len(price),-1,dtype=int)
    if finite.sum()>100:
        rank=stats.rankdata(price[finite],method="ordinal")
        quintile[finite]=np.minimum(4,((rank-1)*5/len(rank)).astype(int))
        for group in range(5):
            use=quintile==group
            if use.sum()<30:continue
            cc=c[use];xx=(matched[use]/.1).ravel()[:,None]
            g=np.repeat(np.arange(len(cc)),16);cu=np.tile(np.arange(16),len(cc));off=np.repeat(np.log(cc.sum(1)),16)
            fit=poisson_fit(cc.ravel(),xx,off,grid=g,cuisine=cu,cluster=g)
            z=fit["coef"][0];rows.append(("housing_quintile_"+str(group+1),z["irr"],z["ci_low"],z["ci_high"],len(cc),fit["converged"]))
    write_csv(out/"model_extensions.csv",["specification","irr","ci_low","ci_high","cells","converged"],rows)
    write_csv(out/"housing_price_quintile_400m.csv",["grid_x","grid_y","housing_log_price","quintile"],[[cell_keys[i][0],cell_keys[i][1],price[i] if np.isfinite(price[i]) else "",int(quintile[i]+1) if quintile[i]>=0 else ""] for i in range(len(price))])
    return rows,quintile


def plot_context(f,context,quintile,extensions,out):
    keys=f["keys"];xy=f["xy"];pop=f["n"]>=30
    names=["non_restaurant_pois","cbd_distance_km","road_density_km_per_km2","building_count","shopping_pois","nighttime_light"]
    fig,axs=plt.subplots(2,3,figsize=(13,8),dpi=180)
    for ax,name in zip(axs.flat,names):
        vals=np.array([context[k].get(name,np.nan) for k in keys],float)
        use=pop&np.isfinite(vals);im=ax.scatter(xy[use,0],xy[use,1],c=vals[use],s=8,marker="s",cmap="viridis")
        ax.set_title(name.replace("_"," "));ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(im,ax=ax,shrink=.7)
    fig.tight_layout();fig.savefig(out/"supp_S8_urban_covariates_400m.png");plt.close(fig)
    xym=f["xy"][f["panel"]]
    if len(xym)==len(quintile) and (quintile>=0).any():
        fig,axs=plt.subplots(1,3,figsize=(14,5),dpi=180)
        price=np.array([context[k].get("housing_log_price",np.nan) for k in keys],float)[f["panel"]]
        use=np.isfinite(price);im=axs[0].scatter(xym[use,0],xym[use,1],c=np.exp(price[use]),s=9,marker="s",cmap="viridis")
        fig.colorbar(im,ax=axs[0],label="CNY / m²");axs[0].set_title("Interpolated area housing price")
        im=axs[1].scatter(xym[use,0],xym[use,1],c=quintile[use]+1,s=9,marker="s",cmap="viridis",vmin=1,vmax=5)
        fig.colorbar(im,ax=axs[1],label="Quintile");axs[1].set_title("Grid price quintile")
        for ax in axs[:2]:ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
        points=[r for r in extensions if r[0].startswith("housing_quintile_")]
        if points:
            xx=np.arange(1,len(points)+1);estimate=np.array([r[1] for r in points])
            axs[2].errorbar(xx,estimate,yerr=[estimate-np.array([r[2] for r in points]),np.array([r[3] for r in points])-estimate],fmt="o",capsize=3)
            axs[2].axhline(1,color="grey",linestyle="--");axs[2].set_xticks(xx)
            axs[2].set_xlabel("Housing-price quintile");axs[2].set_ylabel("IRR per 10 percentage points")
        axs[2].set_title("Matched-origin association")
        fig.tight_layout();fig.savefig(out/"supp_S9_housing_context.png");plt.close(fig)


def heatmap(matrix, path, title, lo=None, hi=None):
    fig,ax=plt.subplots(figsize=(10,8),dpi=180)
    im=ax.imshow(matrix,aspect="auto",cmap="RdBu_r",vmin=lo,vmax=hi)
    ax.set_xticks(range(16));ax.set_xticklabels(CUISINES,rotation=90)
    ax.set_yticks(range(31));ax.set_yticklabels(PROVINCES)
    ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def plot_exposure_figure(st,q,out):
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(11,8),dpi=180)
    values=np.log2(np.maximum(st["matrix"],1e-12))
    im=ax.imshow(values,aspect="auto",cmap="RdBu_r",vmin=-2,vmax=2)
    for i,j in np.argwhere(MATCH==1):ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor="black",linewidth=.7))
    for i,j in np.argwhere(q<.05):ax.plot(j,i,"k.",markersize=2)
    ax.set_xticks(range(16));ax.set_xticklabels(CUISINES,rotation=90)
    ax.set_yticks(range(31));ax.set_yticklabels(PROVINCES,fontsize=7)
    ax.set_title("Origin–cuisine exposure quotient, 400 m")
    fig.colorbar(im,ax=ax,label="log₂ quotient",shrink=.8)
    fig.tight_layout();fig.savefig(out/"fig4_origin_cuisine_exposure.png");plt.close(fig)


def plot_outputs(scales, detail, model, out):
    x=np.array([s["scale_m"] for s in scales])
    fig,axs=plt.subplots(1,3,figsize=(13,4),dpi=180)
    for name,label in [("population_coverage","Residents"),("restaurant_coverage","Restaurants")]:axs[0].plot(x,[s[name] for s in scales],marker="o",label=label)
    axs[0].legend();axs[0].set_ylabel("Coverage")
    for name,label in [("spearman_shannon","Same-cell Spearman"),("bivariate_moran","Bivariate Moran")]:axs[1].plot(x,[s[name] for s in scales],marker="o",label=label)
    axs[1].legend();axs[1].set_ylabel("Association")
    for name,label in [("equal_direct","Equal cuisine"),("weighted_direct","Restaurant weighted")]:axs[2].plot(x,[s[name] for s in scales],marker="o",label=label)
    axs[2].axhline(1,color="grey",linestyle="--");axs[2].legend();axs[2].set_ylabel("Direct quotient")
    for ax in axs:ax.set_xlabel("Grid size (m)");ax.axvline(400,color="grey",alpha=.4)
    fig.tight_layout();fig.savefig(out/"scale_summary.png");plt.close(fig)
    st=detail[400]
    heatmap(np.log2(np.maximum(st["matrix"],1e-12)),out/"origin_cuisine_exposure_400m.png","Log2 origin–cuisine exposure quotient")
    heatmap(MATCH,out/"prespecified_mapping.png","Prespecified origin–cuisine mapping",0,1)
    fig,ax=plt.subplots(figsize=(9,4),dpi=180)
    ax.bar(np.arange(16),st["direct"]);ax.axhline(1,color="black",linestyle="--")
    ax.set_xticks(range(16));ax.set_xticklabels(CUISINES,rotation=90);ax.set_ylabel("Direct quotient")
    fig.tight_layout();fig.savefig(out/"direct_quotients_400m.png");plt.close(fig)
    for size in (300,400,500):
        f=detail[size]["frame"];d=detail[size]
        xy=f["xy"][d["local_sel"]]
        fig,ax=plt.subplots(figsize=(7,7),dpi=180)
        q=np.log2(np.maximum(d["local"],1e-12));im=ax.scatter(xy[:,0],xy[:,1],c=q,s=10,cmap="RdBu_r",vmin=-1,vmax=1,marker="s")
        ax.set_aspect("equal");ax.set_title(f"Local log2 correspondence, {size} m");fig.colorbar(im,ax=ax)
        fig.tight_layout();fig.savefig(out/f"local_correspondence_{size}m.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3),dpi=180)
    for y,fit,label in [(1,model[0],"Pooled"),(0,model[1],"Grid + cuisine FE")]:
        z=fit["coef"][0];ax.errorbar(z["irr"],y,xerr=[[z["irr"]-z["ci_low"]],[z["ci_high"]-z["irr"]]],fmt="o",label=label)
    ax.set_yticks([0,1]);ax.set_yticklabels(["Grid + cuisine FE","Pooled"]);ax.axvline(1,color="grey",linestyle="--");ax.set_xlabel("IRR per 10 percentage points")
    fig.tight_layout();fig.savefig(out/"count_model_available_specs.png");plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,8),dpi=180)
    fields=[("populated_cells","Populated cells"),("joint_cells","Jointly eligible cells"),("population_coverage","Population coverage"),("restaurant_coverage","Restaurant coverage"),("spearman_shannon","Same-cell Spearman"),("bivariate_moran","Bivariate Moran")]
    for ax,(field,title) in zip(axs.flat,fields):
        ax.plot(x,[s[field] for s in scales],"o-",lw=1.2);ax.axvline(400,color="grey",linestyle="--",lw=.7)
        ax.set_title(title);ax.set_xlabel("Grid size (m)")
    fig.tight_layout();fig.savefig(out/"fig3_scale_support_and_association.png");plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(16,4),dpi=180)
    axs[0].bar(range(16),detail[400]["direct"]);axs[0].axhline(1,color="grey",linestyle="--")
    axs[0].set_xticks(range(16));axs[0].set_xticklabels(CUISINES,rotation=90,fontsize=7);axs[0].set_title("Direct correspondence, 400 m")
    for field,label in [("equal_direct","Equal cuisine"),("weighted_direct","Restaurant weighted")]:
        axs[1].plot(x,[s[field] for s in scales],"o-",label=label)
    axs[1].axhline(1,color="grey",linestyle="--");axs[1].legend(fontsize=7);axs[1].set_title("Citywide indices")
    med=[];q1=[];q3=[]
    for size in SCALES:
        value=detail[size]["local"];med.append(np.median(value));q1.append(np.quantile(value,.25));q3.append(np.quantile(value,.75))
    axs[2].plot(x,med,"o-");axs[2].fill_between(x,q1,q3,alpha=.25);axs[2].axhline(1,color="grey",linestyle="--")
    axs[2].set_title("Median local index and IQR")
    for ax in axs[1:]:ax.set_xlabel("Grid size (m)")
    fig.tight_layout();fig.savefig(out/"fig5_direct_correspondence.png");plt.close(fig)


def plot_composition_maps(f,out):
    xy=f["xy"];p=f["p"];c=f["c"]
    fig,axs=plt.subplots(2,2,figsize=(10,9),dpi=180)
    variants=[(p,PROVINCES,"Dominant hukou origin"),(c,CUISINES,"Dominant cuisine"),(p[:,1:],PROVINCES[1:],"Dominant non-Guangdong origin"),(c[:,1:],CUISINES[1:],"Dominant non-Guangdong cuisine")]
    for ax,(arr,names,title) in zip(axs.flat,variants):
        valid=arr.sum(1)>0;dominant=arr[valid].argmax(1)
        ax.scatter(xy[valid,0],xy[valid,1],c=dominant,s=8,marker="s",cmap="tab20",vmin=0,vmax=max(len(names)-1,1))
        ax.set_title(title);ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout();fig.savefig(out/"fig2_dominant_origin_cuisine_maps.png");plt.close(fig)


def plot_model_summary(pooled,demographic,urban,fixed,extensions,out):
    fig,axs=plt.subplots(2,2,figsize=(10,8),dpi=180)
    specs=[("M1 pooled",pooled),("M2 demographic",demographic),("M3 urban",urban),("M4 grid + cuisine FE",fixed)]
    points=[(name,m["coef"][0]) for name,m in specs if m]
    for i,(name,z) in enumerate(points):
        axs[0,0].errorbar(z["irr"],i,xerr=[[z["irr"]-z["ci_low"]],[z["ci_high"]-z["irr"]]],fmt="o")
    axs[0,0].set_yticks(range(len(points)));axs[0,0].set_yticklabels([x[0] for x in points]);axs[0,0].invert_yaxis()
    groups=[("scale_","Adjacent scales",axs[0,1]),("_population","Administrative types",axs[1,0]),("interaction_","Context interactions",axs[1,1])]
    for prefix,title,ax in groups:
        rr=[r for r in extensions if (prefix in r[0] if prefix=="_population" else r[0].startswith(prefix))]
        for i,r in enumerate(rr):ax.errorbar(r[1],i,xerr=[[r[1]-r[2]],[r[3]-r[1]]],fmt="o")
        ax.set_yticks(range(len(rr)));ax.set_yticklabels([r[0].replace("interaction_","") for r in rr],fontsize=8)
        ax.invert_yaxis();ax.set_title(title)
    axs[0,0].set_title("Sequential models")
    for ax in axs.flat:ax.axvline(1,color="grey",linestyle="--",lw=.7);ax.set_xlabel("Incidence-rate ratio")
    fig.tight_layout();fig.savefig(out/"fig7_count_models_and_heterogeneity.png");plt.close(fig)


def plot_hubei_density(base,a,out):
    gx=np.array([k[0] for k in base]);gy=np.array([k[1] for k in base])
    left,right=gx.min(),gx.max();bottom,top=gy.min(),gy.max()
    residents=np.zeros((top-bottom+1,right-left+1));restaurants=np.zeros_like(residents)
    for (x,y),cell in base.items():
        residents[y-bottom,x-left]=cell["p"][2];restaurants[y-bottom,x-left]=cell["c"][2]
    fig,axs=plt.subplots(1,2,figsize=(12,5),dpi=180)
    for ax,arr,title in [(axs[0],residents,"Hubei-registered residents"),(axs[1],restaurants,"Hubei-cuisine restaurants")]:
        smooth=gaussian_filter(arr,sigma=5)
        ax.imshow(smooth,origin="lower",extent=[a.x_origin+left*100,a.x_origin+(right+1)*100,a.y_origin+bottom*100,a.y_origin+(top+1)*100],cmap="magma")
        ax.set_title(title+" (500 m Gaussian bandwidth)");ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout();fig.savefig(out/"fig1_hubei_density_panels.png");plt.close(fig)


def main():
    a=args_parser()
    if a.prepare_variables or a.prepare_only or (not a.skip_context and not (a.variables_dir/"source_manifest.json").exists()):
        print("Clipping contextual variables to the study boundary.",flush=True)
        prepare_variables(a)
    if a.prepare_only:
        export_analysis_grids(a)
        print("Prepared variables only; no point analysis was rerun.")
        return
    a.out.mkdir(parents=True,exist_ok=True)
    print("Reading point GDBs; input files are read-only.",flush=True)
    base,rest,summary=read_points(a)
    rng=np.random.RandomState(a.seed)
    scales=[];detail={}
    for size in SCALES:
        print(f"Grid {size} m",flush=True)
        f=frame(aggregate(base,size),size,a)
        st=direct_stats(f)
        st["frame"]=f;detail[size]=st
        scales.append(save_scale(f,st,size,a.out,a))
    print("Direct-label permutations",flush=True)
    for size in SCALES:
        detail[size]["permutation"]=permute_direct(detail[size],a.permutations if size==400 else a.scale_permutations,rng)
    context=None
    if not a.skip_context:
        print("Aggregating clipped urban covariates and housing context",flush=True)
        context=read_context(a,400,detail[400]["frame"]["keys"])
        f400=detail[400]["frame"]
        context_names=["non_restaurant_pois","shopping_pois","cbd_distance_km","metro_distance_km","road_density_km_per_km2","building_count","commercial_area_proxy","nighttime_light","housing_log_price"]
        write_csv(a.out/"urban_covariates_400m.csv",["grid_x","grid_y",*context_names],[[*k,*[context[k].get(n,"") for n in context_names]] for k in f400["keys"]])
    print("Poisson count models",flush=True)
    pooled,demographic,urban,fixed,panel=regression(detail[400]["frame"],context)
    write_csv(a.out/"scale_metrics.csv",list(scales[0]),[[s[k] for k in scales[0]] for s in scales])
    for size in SCALES:
        st=detail[size]
        write_csv(a.out/f"direct_quotients_{size}m.csv",["cuisine","quotient","restaurant_count","permutation_p","BH_q"],[[CUISINES[j],st["direct"][j],int(st["c"].sum(0)[j]),st["permutation"]["direct_p"][j],st["permutation"]["direct_q"][j]] for j in range(16)])
    write_csv(a.out/"exposure_matrix_400m.csv",["hukou_province",*CUISINES],[[PROVINCES[i],*detail[400]["matrix"][i]] for i in range(31)])
    if a.matrix_permutations:
        print("Exposure-matrix two-sided label permutations",flush=True)
        mp,mq=matrix_permutation(detail[400],a.matrix_permutations,rng)
        write_csv(a.out/"exposure_matrix_p_400m.csv",["hukou_province",*CUISINES],[[PROVINCES[i],*mp[i]] for i in range(31)])
        write_csv(a.out/"exposure_matrix_BH_q_400m.csv",["hukou_province",*CUISINES],[[PROVINCES[i],*mq[i]] for i in range(31)])
        plot_exposure_figure(detail[400],mq,a.out)
    write_csv(a.out/"local_correspondence_400m.csv",["grid_x","grid_y","local_index"],[[*detail[400]["frame"]["keys"][i],q] for i,q in zip(np.where(detail[400]["local_sel"])[0],detail[400]["local"])])
    plot_outputs(scales,detail,(pooled,fixed),a.out)
    plot_composition_maps(detail[400]["frame"],a.out)
    plot_hubei_density(base,a,a.out)
    plot_additional(scales,detail,(pooled,demographic,urban,fixed),panel,a.out,rng,a.scale_permutations)
    print("Supplementary variograms and source variants",flush=True)
    variogram_and_thresholds(detail,a.out,rng)
    variants=source_sensitivity(detail[400]["frame"],rest,a,a.out)
    extensions=[]
    if context is not None:
        print("Adjacent-scale, administrative-type, interaction and housing models",flush=True)
        extensions,quintile=model_extensions(detail,context,a.out)
        plot_context(detail[400]["frame"],context,quintile,extensions,a.out)
    plot_model_summary(pooled,demographic,urban,fixed,extensions,a.out)
    expected={"eligible_population":11375276,"joint_cells_400m":1073,"model_cells_400m":1563,"equal_direct_400m":1.380,"fixed_effects_irr":1.264}
    observed={"eligible_population":summary["population_eligible"],"joint_cells_400m":scales[3]["joint_cells"],"model_cells_400m":scales[3]["model_cells"],"equal_direct_400m":scales[3]["equal_direct"],"fixed_effects_irr":fixed["coef"][0]["irr"]}
    audit={k:{"manuscript":expected[k],"computed":observed[k],"difference":observed[k]-expected[k]} for k in expected}
    report={"input_summary":summary,"scale_metrics":scales,"manuscript_reproduction_audit":audit,"permutation_400m":{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in detail[400]["permutation"].items()},"model_pooled":{k:v for k,v in pooled.items() if k!='pred'},"model_demographic":{k:v for k,v in demographic.items() if k!='pred'} if demographic else None,"model_urban":{k:v for k,v in urban.items() if k!='pred'} if urban else None,"model_fixed_effects":{k:v for k,v in fixed.items() if k!='pred'},"source_sensitivity":variants,"model_extensions":extensions,"not_reproducible_exactly_from_supplied_sources":["original Baidu/Dianping ingestion and 75-m fusion (restaurant GDB is already fused)","original positional correction (population GDB is already corrected)","published grid origin and all original covariate transformations","original commercial-area polygons; nearby shopping POIs are used as a labelled proxy","original housing variogram and holdout split; local kriging is reconstructed","Fig. 1 photographs, external basemaps and national flow map"]}
    (a.out/"analysis_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if not a.skip_context:
        export_analysis_grids(a)
    if a.limit_pop:
        print("SMOKE TEST ONLY: --limit-pop was used; do not cite these outputs.")
    print(f"Finished: {a.out.resolve()}")


if __name__=="__main__":
    main()
