from shelter import setup_system



def create_system_page(system_name, filepath):
    system = setup_system(system_name, candidate_data=True, table_id='pscomppars')
    system.to_obsidian(filepath)

filepath = "/Users/trm143@student.bham.ac.uk/Documents/Obsidian/Obsidian Notes/Notes/30 - Research/33 - PhD/33.4 - Notes/Planetary Systems"

create_system_page('TOI-6690', filepath)