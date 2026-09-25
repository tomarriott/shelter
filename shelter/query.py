def query_simbad(system, name=None):

    def _get_id(all_ids, substr):
        try:
            return list(filter(lambda x: substr in x, all_ids))[0].lstrip(substr)
        except Exception:
            return None

    from astroquery.simbad import Simbad

    if name is None:
        name = system.name

    try:
        position_table = Simbad.query_object(name)
        id_table = Simbad.query_objectids(name)
    except Exception as e:
        print(f'Unable to resolve {name} with Simbad: {e}')
        return system

    system.rastr = position_table['RA'].value.data[0]
    system.decstr = position_table['DEC'].value.data[0]

    all_ids = [item.decode() for item in id_table['ID'].value]

    system.tic = _get_id(all_ids, 'TIC ')
    system.toi = _get_id(all_ids, 'TOI-')
    system.gaia = _get_id(all_ids, 'Gaia DR3 ')
    system.twomass = _get_id(all_ids, '2MASS ')
    system.wise = _get_id(all_ids, 'WISEA ')

    return system

def query_tic(star, tic=None):
    from astroquery.mast import Catalogs

    if tic is None:
        if star.tic is None:
            star = query_simbad(star)
        if star.tic is None:
            print(f'Unable to get TIC ID for {star.name}.')
            return star
        tic = star.tic

    result_table = Catalogs.query_criteria(catalog='TIC', ID=tic)

    star.ra = result_table['ra'].value[0]
    star.dec = result_table['dec'].value[0]

    for mag in ['Bmag', 'Vmag', 'umag', 'gmag', 'rmag', 'imag', 'zmag', 'Jmag', 'Hmag', 'Kmag', 'Tmag', 'GAIAmag', 'w1mag', 'w2mag', 'w3mag', 'w4mag']:
        star.set_param(mag, result_table[mag].value[0], result_table[f'e_{mag}'].value[0])

    star.set_param('mass', result_table['mass'].value[0], result_table['e_mass'].value[0])
    star.set_param('radius', result_table['rad'].value[0], result_table['e_rad'].value[0])
    star.set_param('temperature', result_table['Teff'].value[0], result_table['e_Teff'].value[0])
    star.set_param('density', result_table['rho'].value[0], result_table['e_rho'].value[0])
    star.set_param('logg', result_table['logg'].value[0], result_table['e_logg'].value[0])
    star.set_param('luminosity', result_table['lum'].value[0], result_table['e_lum'].value[0])
    star.set_param('distance', result_table['d'].value[0], result_table['e_d'].value[0])

    return star