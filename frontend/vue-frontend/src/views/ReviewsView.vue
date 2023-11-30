<template>
  <div>
    <h1>Get Recommendation</h1>
    <div class="form-group">
      <button @click="getRecommendation">Get Recommendation</button>
      <input type="text" id="recommendation" v-model="recommendation">
    </div>
    <div class="form-group">
      <label for="locationSelect">Select Location:</label>
      <select id="locationSelect" v-model="selectedLocation">
        <option v-for="(location, index) in locations" :key="index" :value="location">
          {{ location.name }}
        </option>
      </select>
    </div>

    <div class="form-group">
      <label for="reviewInput">Give Review (1-5):</label>
      <input
        type="number"
        id="reviewInput"
        v-model="rating"
        min="1"
        max="5"
      />
    </div>

    <div class="form-group">
      <button @click="postReview">Post Review</button>
    </div>
  </div>
</template>

<script>
  import axios from 'axios';

  export default {
    name: 'ReviewsView',
    data() {
      return {
        locations: [], // Initialize an array to store locations
        selectedLocation: "",
        rating: "",
        recommendation: ""
      };
    },
    mounted() {
      this.fetchLocations(); // Call the function to fetch locations when the component is mounted
    },
    methods: {
      async fetchLocations() {
        try {
          const response = await axios.get('http://127.0.0.1:5000/locations'); // Make a GET request to /locations using Axios
          this.locations = response.data; // Update the locations array with the retrieved data
        } catch (error) {
          console.error('Error fetching locations:', error);
        }
      },
      async getRecommendation() {
        try {
          const response = await axios.get('http://127.0.0.1:5000/recommendation'); // Make a GET request to /locations using Axios
          this.recommendation = response.data
        } catch (error) {
          console.error('Error fetching recommendation:', error);
        }
      },
      async postReview() {
        await axios.post('http://127.0.0.1:5000/reviews', {
          username: localStorage.getItem('currentUser'),
          location: this.selectedLocation.name,
          rating: this.rating
        });
      }
    }
  };
</script>
  
<style scoped>
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background-color: #f4f4f4;
  }
  .login-container {
    background-color: white;
    padding: 20px;
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  }
  .form-group {
    margin-bottom: 15px;
  }
  .form-group label {
    display: block;
    font-weight: bold;
    margin-bottom: 5px;
  }
  .form-group input {
    width: 100%;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
  .form-group button {
    padding: 8px 12px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  .form-group button:hover {
    background-color: #0056b3;
  }
</style>
  