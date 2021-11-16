<template>
    <div class="search">
        <i class="search__btn" @click="search">
            <svg id="magnify" viewBox="0 0 51.38 51.38">
                <circle cx="18.5" cy="18.5" r="16" />
                <line x1="29.81" y1="29.81" x2="49.61" y2="49.61" />
            </svg>
        </i>
        <input
            type="text"
            class="search__input"
            placeholder="Search..."
            v-model="query"
            v-on:keyup.enter="search"
        />
    </div>
</template>

<script>
export default {
    name: "Search",
    data() {
        return {
            query: "",
        };
    },
    methods: {
        search() {
            this.$emit("predict", this.query);
            this.query = "";
        },
    },
};
</script>

<style scoped lang="scss">
.search {
    position: relative;
    width: 500px;
    height: 50px;

    &:hover &__input {
        background: #f8f8f8;
        box-shadow: inset 5px 5px 5px #d8d8d8, inset -5px -5px 5px #ffffff;
    }

    &__input {
        position: absolute;
        z-index: 1;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        padding: 5px 20px;
        border-radius: 25px; // half of height

        border: none;
        outline: none;
        font-size: 1rem;
        background: #f8f8f8;
        box-shadow: 5px 5px 5px #d8d8d8, -5px -5px 5px #ffffff;

        &:focus,
        &:active {
            background: #f8f8f8;
            box-shadow: inset 5px 5px 5px #d8d8d8, inset -5px -5px 5px #ffffff;
        }
    }

    &__btn {
        position: absolute;
        z-index: 2;
        right: 0;
        height: 100%;
        width: 50px; // equal to height of search bar

        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;

        #magnify {
            height: 20px;
            width: 20px;
            fill: none;
            stroke: #2c3e50;
            stroke-width: 5px;
        }
    }
}
</style>