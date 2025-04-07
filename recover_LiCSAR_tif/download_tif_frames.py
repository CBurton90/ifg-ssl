from licsar_web_tools_downloading import download_LiCSAR_portal_data

#%% An example on the North Anatolian fault

frameID = '014A_04939_131313'                                       # As defined on the LiCSAR portal
date_start = 20190901                                               # YYYYMMDD
date_end   = 20200101                                               # YYYYMMDD
download_metadata = True                                            # the network of baselines, DEM etc.  
between_epoch_files = ['geo.unw.png', 'geo.cc.png']                 # Possible files: 'geo.cc.png', 'geo.cc.tif', 'geo.diff.png', 'geo.diff_pha.tif', 'geo_diff_unfiltered.png', 'geo_diff_unfiltered_pha.tif', 'geo.unw.png', 'geo.unw.tif'
epoch_files = ['geo.mli.png','ztd.jpg']                             # Possible files: 'geo.mli.png', 'geo.mli.tif', 'sltd.geo.tif', 'ztd.geo.tif', 'ztd.jpg'
n_para = 4                                                          # Parallelisation.  The number of cores is a good starting point.    


# download_LiCSAR_portal_data(frameID, date_start, date_end, download_metadata, epoch_files, between_epoch_files, n_para)

def create_tifs(annonation_path):

    frame_list = []

    annotations_list = os.listdir(annotation_path)

    for idx, annotation_file in tqdm(enumerate(annotations_list)):
        annotation = json.load(open(annotation_path + annotation_file, "r"))

        if "Deformation" in annotation["label"]:
            frame = {
                'frameID': annotation["frameID"],
                'primary_date': annotation['primary_date'],
                'secondary_date': annotation['secondary_date']
                }

            frame_list.append(frame)

    for f in frame_list:
        download_metadata = False
        epoch_files = None
        between_epoch_files = ['geo.diff_pha.tif']
        download_LiCSAR_portal_data(f['frameID'], f['primary_date'], f['secondary_date'], download_metadata, epoch_files, between_epoch_files, 4)