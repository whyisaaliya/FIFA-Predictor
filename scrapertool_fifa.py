import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# We must send a User-Agent header to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36'
}

def get_soup(url):
    """Fetches a URL and returns a BeautifulSoup object."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Raises an exception for bad status codes
        return BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        # We'll print errors only if the page itself fails to load
        print(f"Error fetching {url}: {e}") 
        return None

def scrape_player_stats():
    """
    Scrapes the top scorer (player stats) list for all World Cups
    from 1994 to 2022.
    """
    base_url = (
        "https://www.transfermarkt.co.in/world-cup/scorerliste/"
        "pokalwettbewerb/FIWC/saison_id/{year}/altersklasse/alle/plus/{page}"
    )
    
    world_cup_years = [1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
    saison_ids = [year - 1 for year in world_cup_years]
    
    all_players_data = []

    for year in saison_ids:
        wc_year = year + 1
        page = 1
        
        while True:
            url = base_url.format(year=year, page=page)
            soup = get_soup(url)
            if not soup:
                break 

            table = soup.find('table', class_='items')
            if not table:
                break

            rows = table.find('tbody').find_all('tr')
            
            if not rows:
                break
                
            for row in rows:
                try:
                    cells = row.find_all('td')
                    
                    # --- THIS IS THE FIX ---
                    # If the row doesn't have at least 11 columns, 
                    # it's not a player row. Skip it.
                    if len(cells) < 11:
                        continue 
                    # ---------------------

                    rank = cells[0].text.strip()
                    player_name = cells[1].find('a', class_='spielprofil_tooltip').text.strip()
                    club = cells[2].find('img')['title'].strip()
                    nationality = cells[3].find('img', class_='flaggenrahmen')['title'].strip()
                    age = cells[4].text.strip()
                    games = cells[5].text.strip()
                    assists = cells[6].text.strip()
                    sub_off = cells[7].text.strip()
                    goals = cells[8].text.strip()
                    sub_on = cells[9].text.strip()
                    points = cells[10].text.strip()

                    all_players_data.append({
                        'World_Cup_Year': wc_year,
                        'Rank': rank,
                        'Player': player_name,
                        'Club': club,
                        'Nationality': nationality,
                        'Age': age,
                        'Games': games,
                        'Assists': assists,
                        'Sub_Off': sub_off,
                        'Goals': goals,
                        'Sub_On': sub_on,
                        'Points': points
                    })
                except Exception as e:
                    # If a row fails parsing, just log it quietly and move on
                    # print(f"Skipping a problematic row for WC {wc_year}: {e}")
                    pass # Silently skip the row
                    
            page += 1

    print("Player stats scraping finished.")
    df = pd.DataFrame(all_players_data)
    return df

def scrape_team_stats():
    """
    Scrapes the all-time World Cup table (team stats).
    """
    url = ("https://www.transfermarkt.co.in/weltmeisterschaft/"
           "ewigeTabelle/pokalwettbewerb/FIWC")
    
    soup = get_soup(url)
    if not soup:
        print("Failed to fetch team stats page.")
        return pd.DataFrame()

    all_teams_data = []
    table = soup.find('table', class_='items')
    
    if not table:
        print("Could not find the team stats table.")
        return pd.DataFrame()

    rows = table.find('tbody').find_all('tr')

    for row in rows:
        try:
            cells = row.find_all('td')

            # --- ADDING A FIX HERE TOO ---
            if len(cells) < 8:
                continue
            # ---------------------------
            
            rank = cells[0].text.strip()
            country = cells[1].find('a').text.strip() 
            matches = cells[2].text.strip()
            wins = cells[3].text.strip()
            draws = cells[4].text.strip()
            losses = cells[5].text.strip()
            goal_diff = cells[6].text.strip()
            points = cells[7].text.strip()

            all_teams_data.append({
                'Rank': rank,
                'Country': country,
                'Matches': matches,
                'W': wins,
                'D': draws,
                'L': losses,
                'GD': goal_diff,
                'Points': points
            })
        except Exception as e:
            # print(f"Skipping a problematic team row: {e}")
            pass # Silently skip the row

    print("Team stats scraping finished.")
    df = pd.DataFrame(all_teams_data)
    return df

def main():
    """
    Main function to run both scrapers and save data to CSV files.
    """
    
    # --- Part 1: Scrape Player Stats ---
    player_stats_df = scrape_player_stats()
    
    if not player_stats_df.empty:
        player_stats_df.to_csv('transfermarkt_player_stats_1994-2022.csv', 
                               index=False)
        print("\nSuccessfully saved player stats to "
              "'transfermarkt_player_stats_1994-2022.csv'")
        print(player_stats_df.head())
    else:
        print("\nNo player data was scraped.")

    # --- Part 2: Scrape Team Stats ---
    team_stats_df = scrape_team_stats()
    
    if not team_stats_df.empty:
        team_stats_df.to_csv('transfermarkt_team_stats_all-time.csv', 
                            index=False)
        print("\nSuccessfully saved all-time team stats to "
              "'transfermarkt_team_stats_all-time.csv'")
        print(team_stats_df.head())
    else:
        print("\nNo team data was scraped.")

if __name__ == "__main__":
    main()