<template>
    <main class="page-content">
        <h1 class="title">Sentavis</h1>
        <div class="tweet-container">
            <search-tag @predict="predict_tag" />
        </div>
        <div class="usertext-container">
            <user-text @predict="predict_text" />
        </div>
        <div class="card-container">
            <card
                v-for="text in texts"
                :key="text.id"
                :body="text.body"
                :sentiment="text.sentiment"
            />
        </div>
    </main>
</template>

<script>
import SearchTag from "./components/SearchTag.vue";
import UserText from "./components/UserText.vue";
import Card from "./components/Card.vue";
export default {
    name: "Sentavis",
    components: { SearchTag, UserText, Card },
    data() {
        return {
            texts: [],
        };
    },
    methods: {
        fetch_predictions(url, formData) {
            fetch(url, {
                method: "POST",
                body: formData,
                mode: "cors",
            })
                .then((response) => response.json())
                .then((data) => {
                    data.forEach((text) => {
                        console.log(text.body);
                        console.log(text.sentiment);
                        this.texts.unshift({
                            id: Math.floor(Math.random() * 100000),
                            body: text.body,
                            sentiment: text.sentiment,
                        });
                    });
                });
        },
        predict_text(text) {
            const formData = new FormData();
            formData.append("text", text);
            this.fetch_predictions(
                "http://127.0.0.1:5000/predict_text/",
                formData
            );
        },
        predict_tag(tag) {
            const formData = new FormData();
            formData.append("tag", tag);
            this.fetch_predictions(
                "http://127.0.0.1:5000/predict_tag/",
                formData
            );
        },
    },
};
</script>

<style lang="scss" src="./main.scss"/>
