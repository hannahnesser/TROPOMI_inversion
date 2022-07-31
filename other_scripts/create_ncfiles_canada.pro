;A program to create Mexico and Canada emissions grids
pro create_ncfiles_canada
;=============
;mexico
;==============
;filesector=['1A','1B'];,'2B','3A','3C','4A','4B','4C','4D']
;nfile = n_elements(filesector)

;--read header and Area
filename = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/prior/canada/Area_Global.nc'
Cdfid=NCDF_OPEN(filename)
NCDF_VARGET, Cdfid ,'lat' ,lat
NCDF_VARGET, Cdfid ,'lon' ,lon
NCDF_VARGET, Cdfid ,'cell_area',cell_area
NCDF_CLOSE,Cdfid


finalsector = ['coal','oil1','oil2','oil3','gas1','gas2','gas3',$
		'livestock_A','livestock_B','landfill','waste','other_anthro1','other_anthro2','biomass']
N_GCsector = n_elements(finalsector)

nlon=n_elements(lon)
nlat=n_elements(lat)

emisdata1=fltarr(nlon,nlat,N_GCsector)

for isec = 0, N_GCsector-1 do begin

	thissec = finalsector[isec]
	if thissec eq 'coal' then begin 
	filesec='1B1'
	varsec=['1B1_total']
	endif

        if thissec eq 'oil1' then begin
        filesec=['1B2a']
	varsec=['1B2a_total']
	endif

	if thissec eq 'oil2' then begin
	filesec=['1B2ci']
        varsec=['1B2ci1 (oil production)',$
                '1B2ci1 (oil sands)', '1B2ci1 (oil transport)',$
                '1B2ci3 (combined)']
        endif

	if thissec eq 'oil3' then begin
	filesec=['1B2cii']
	varsec=['1B2cii1 (oil production)','1B2cii1 (oil refining)',$
                '1B2cii1 (oil sands)', '1B2cii1 (oil transport)',$
                '1B2cii3 (combined)']
	endif
	
        if thissec eq 'gas1' then begin
        filesec=['1B2b']
        varsec=['1B2b_total']
        endif

        if thissec eq 'gas2' then begin
        filesec=['1B2ci']
        varsec = ['1B2ci2 (gas distribution)', '1B2ci2 (gas processing)',$
                  '1B2ci2 (gas production)', '1B2ci2 (gas storage)',$
                  '1B2ci2 (gas transmission)']
        endif

        if thissec eq 'gas3' then begin
        filesec=['1B2cii']
        varsec = ['1B2cii2 (gas distribution)', '1B2cii2 (gas processing)',$
                  '1B2cii2 (gas production)', '1B2cii2 (gas storage)',$
                  '1B2cii2 (gas transmission)']
        endif


	if thissec eq 'livestock_A' then begin
	filesec='3A'
	varsec=['3A_total']
	endif

	if thissec eq 'livestock_B' then begin
	filesec='3B'
	varsec=['3B_total']
	endif

	if thissec eq 'landfill' then begin
	filesec=['5']
	varsec=['5A1a','5A2','5B1a','5C1']
	endif

	if thissec eq 'waste' then begin
	filesec=['5']
	varsec=['5D1','5D2']
	endif

        if thissec eq 'other_anthro1' then begin
        filesec=['1A']
        varsec=['1A_total']
        endif

        if thissec eq 'other_anthro2' then begin
        filesec=['2']
        varsec=['2_total']
        endif

        if thissec eq 'biomass' then begin
        filesec=['3F']
        varsec=['3F_total']
        endif

	;if thissec eq 'rice' then begin
	
        ;endif


	;NC files
	nsecfile=n_elements(filesec)
	for ifile = 0,nsecfile-1 do begin
		filename = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/prior/canada/can_emis_'+filesec[ifile]+'_2018.nc'
		Cdfid=NCDF_OPEN(filename)
		nvars=n_elements(varsec)
		for ivar = 0 , nvars-1 do begin

			;if thissec ne 'landfill' and thissec ne 'other_anthro' then begin
			NCDF_VARGET, Cdfid ,'emis_ch4_'+varsec[ivar],tmpvar
			;endif else begin
			;NCDF_VARGET, Cdfid ,'emis_ch4_'+filesec[ifile]+'_'+varsec[ivar],tmpvar
			;endelse

			if ivar+ifile eq 0 then begin
			totalvar = tmpvar
			endif else begin
			totalvar = totalvar + tmpvar		
			endelse
		endfor
		NCDF_CLOSE,Cdfid
	endfor

	print, thissec,total(totalvar*cell_area*1e-6)*1e-3,' Gg'
	
	;if isec eq 0 then begin
	;totalemis=total(totalvar*cell_area*1e-6)*1e-3
	;endif else begin
	;totalemis=totalemis+total(totalvar*cell_area*1e-6)*1e-3
	;endelse

	emisdata1(*,*,isec) = totalvar
endfor

;------------
;aggregate to GC sectors
;------------
finalsector = ['coal','oil','gas',$
                'livestock_A','livestock_B','landfill','waste','other_anthro']

N_GCsector = n_elements(finalsector)

emisdata=fltarr(nlon,nlat,N_GCsector)

emisdata(*,*,0) = emisdata1(*,*,0) ;coal
emisdata(*,*,1) = total(emisdata1(*,*,[1,2,3]),3);oil
emisdata(*,*,2) = total(emisdata1(*,*,[4,5,6]),3);gas
emisdata(*,*,3) = emisdata1(*,*,7);livestock_A
emisdata(*,*,4) = emisdata1(*,*,8);livestock_B
emisdata(*,*,5) = emisdata1(*,*,9);landfill
emisdata(*,*,6) = emisdata1(*,*,10);waste
emisdata(*,*,7) = total(emisdata1(*,*,[11,12,13]),3);other_anthro

print,'-----'
for isec = 0, N_GCsector-1 do begin
	print,finalsector[isec],total(emisdata(*,*,isec)*cell_area*1e-6)*1e-3,' Gg'

        if isec eq 0 then begin
        totalemis=total(emisdata(*,*,isec)*cell_area*1e-6)*1e-3
        endif else begin
        totalemis=totalemis+total(emisdata(*,*,isec)*cell_area*1e-6)*1e-3
        endelse
endfor
	print, 'total ',totalemis,' Gg'

	CF=1.0/16.0*6.02e23/(365*24*3600.0)/(1e4)
	emisdata = emisdata * CF
;stop
;--------
;write to NC file
;--------
indx = where(lon ge -180 and lon le 0)
indy = where(lat ge 0 and lat le 90)

lon=lon[indx]
lat=lat[indy]
emisdata=reform(emisdata(indx,indy,*))

donc=1
if (donc) then begin

        for isec = 0, N_GCsector-1 do begin
        thissec = finalsector[isec]

        filename='/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/prior/canada/processed/CAN_Tia2020_'+thissec+'_2018.nc'

        ;open a netcdf file
        FileId  = NCDF_Create( FileName, /Clobber )
        NCDF_Control, FileID, /NoFill

        ;------Definition
        N_Lon  = n_elements(lon)
        iID     = NCDF_DimDef( FileID, 'lon', N_Lon  )
        LonID   = NCDF_VarDef( FileID, 'lon', [iID], /Double  )

        N_Lat = n_elements(lat)
        jID     = NCDF_DimDef( FileID, 'lat', n_Lat )
        LatID  = NCDF_VarDef( FileID, 'lat', [jID], /Double )

        ;------Attribution-----------------
        NCDF_AttPut, FileID, lonID,'long_name','longitude'
        NCDF_AttPut, FileID, lonID,'units','degrees_east'
        NCDF_AttPut, FileID, latID,'long_name','latitude'
        NCDF_AttPut, FileID, latID,'units','degrees_north'
        NCDF_Control, FileID, /EnDef
        ;----------------------------------
        NCDF_VarPut, FileID, lonID, lon, Count=[N_Lon]
        NCDF_VarPut, FileID, latID, lat, Count=[N_Lat]
        ;----------------------------------
        NCDF_Control, FileID, /ReDef
        VarID = NCDF_VarDef( FileID,'emis_ch4', [iID,jID], /Float )
        NCDF_AttPut, FileID, VarID, 'long_name',thissec
        NCDF_AttPut, FileID, VarID,'units','molec/cm2/s'
        ; Switch into data mode
        NCDF_Control, FileID, /EnDef
        NCDF_VarPut, FileID, VarID,emisdata(*,*,isec), Count=[N_Lon,N_Lat]
        NCDF_Close, FileID
        endfor
endif;if do nc



stop




end
