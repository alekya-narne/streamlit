mkdir -p ~/.streamlit/
echo "
[general]n
email = "an908@scarletmail.rutgers.edu"n
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truen
enableCORS=falsen
port = $PORTn
" > ~/.streamlit/config.toml
