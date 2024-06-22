import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Button, Appbar, Card } from 'react-native-paper'; 
import AsyncStorage from '@react-native-async-storage/async-storage'
import MainPage from './MainPage'; 


const HomeScreen = ({ navigation }) => {
    const [userEmail, setUserEmail] = React.useState(null);
  
    React.useEffect(() => {
      const fetchEmail = async () => {
        const email = await AsyncStorage.getItem('userEmail');
        setUserEmail(email);
      };
  
      fetchEmail();
    }, []);
  
    
  
    return (
      <View style={styles.container}>
        {userEmail ? (
          <MainPage />
        ) : (
          <View>
            <Appbar.Header>
              <Appbar.Content title="eBabySitter" />
            </Appbar.Header>
            <View style={styles.content}>
              <Card style={styles.card}>
                <Card.Actions style={styles.cardActions}>
                  <Button mode="contained" onPress={() => navigation.navigate('SignUp')} style={styles.button}>
                    Sign Up
                  </Button>
                  <Button mode="contained" onPress={() => navigation.navigate('Login')} style={styles.button}>
                    Log In
                  </Button>
                </Card.Actions>
              </Card>
            </View>
          </View>
        )}
      </View>
    );
  };
  
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f0f0f0',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  card: {
    width: '100%',
    marginBottom: 20,
    borderRadius: 10,
    elevation: 3,
  },
  cardActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  button: {
    flex: 1,
    height: 60,
    marginHorizontal: 10,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 10,
  },
});

export default HomeScreen;