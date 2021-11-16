<template>
    <div class="user-search">
        <label class="user-search__label" for="user-search__text">
            <span> Enter text to analyze: </span>
            <textarea
                type="text"
                id="user-search__text"
                class="user-search__text"
                placeholder="Enter your text..."
                v-model="text"
            />
        </label>
        <button class="user-search__btn" v-on:click="predict">
            Predict Sentiment
        </button>
    </div>
</template>

<script>
export default {
    name: "UserSearch",
    data() {
        return {
            text: "",
        };
    },
    methods: {
        predict() {
            if (this.text.length > 0) {
                this.$emit("predict", this.text);
            }
            this.text = "";
        },
    },
};
</script>


<style scoped lang="scss">
.user-search {
    position: relative;
    width: 600px;
    padding: 20px;
    margin: 20px;
    border-radius: 15px;

    display: flex;
    flex-direction: column;
    align-items: center;

    background: #f8f8f8;
    box-shadow: 5px 5px 5px #d8d8d8, -5px -5px 5px #ffffff;

    &__label {
        width: 100%;
        height: 100%;

        span {
            display: block;
            padding: 0 15px;
            margin-bottom: 5px;
            color: #2c3e50;
        }
    }

    &__text {
        font-family: Poppins, sans-serif;
        font-size: 0.9rem;
        width: 100%;
        height: 100%;
        padding: 15px;
        resize: vertical;
        overflow: auto;
        color: #3e3e3e;

        border: none;
        border-radius: 15px;
        background: #f8f8f8;
        box-shadow: inset 5px 5px 5px #d8d8d8, inset -5px -5px 5px #ffffff;
    }

    &__btn {
        appearance: none;
        height: 50px;
        width: 175px;
        border-radius: 25px;
        border: none;
        outline: none;
        margin-top: 20px;
        background: #f8f8f8;
        box-shadow: 5px 5px 5px #d8d8d8, -5px -5px 5px #ffffff;
        padding: 5px;
        font-family: Poppins, sans-serif;
        font-weight: 600;
        color: #2c3e50;

        &:hover,
        &:focus {
            background: #f8f8f8;
            box-shadow: inset 5px 5px 5px #d8d8d8, inset -5px -5px 5px #ffffff;
        }
    }
}
</style>