from streamlit_authenticator.utilities.hasher import Hasher
c
hashed_passwords = Hasher(['ayush123']).generate()
print(hashed_passwords)

